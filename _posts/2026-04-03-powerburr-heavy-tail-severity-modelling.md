---
layout: post
title: "The PowerBurr Distribution: When Burr XII Runs Out of Tail"
date: 2026-04-03
categories: [research, severity, insurance-pricing]
author: "Burning Cost"
description: "Liu & Meng's PowerBurr adds a fourth parameter to Burr XII that decouples body shape from tail heaviness. We explain the coupling trap, what the fix actually does, and when UK pricing teams should care."
---

See also our earlier coverage of [the PowerBurr distribution for heavy-tail severity modelling](/2026/04/02/powerburr-severity-modelling-heavy-tails/).

Here is a problem you will recognise if you have ever fitted a severity model to motor BI large-loss data. Your Burr XII has won on AIC. The body fit is fine. Then you look at the top twenty claims — the PPO awards, the catastrophic spinal injuries priced under Ogden tables at -0.25% — and they all sit above the 45-degree line on the QQ plot. You try reducing α to fatten the tail. The extreme fit improves marginally, but now the 60th-to-90th percentile range is wrong, and your layer costs for the £100k xs £50k corridor shift by 7%. You have traded one problem for another.

This is not bad luck. It is structural.

Liu & Meng's *PowerBurr Regression Model for Heavy-Tailed Loss Data and its Application* (*Insurance: Mathematics and Economics* Vol. 127, March 2026) directly targets this constraint.

## Why Burr XII Couples Body and Tail

The Burr XII survival function is:

$$S(x;\, \alpha, \delta, \beta) = \left(1 + \left(\frac{x}{\beta}\right)^\delta\right)^{-\alpha}$$

The tail decays as $$x^{-\alpha\delta}$$, so the tail index is $$\alpha\delta$$. Mean exists iff $$\alpha\delta > 1$$; variance iff $$\alpha\delta > 2$$.

The problem is δ. It does two jobs:

1. It controls the body shape. The mode (when $$\delta > 1$$) is $$\beta \cdot [(\delta-1)/(\alpha\delta+1)]^{1/\delta}$$. Want a sharp peak at £15k? δ needs to be roughly 1.6–2.0.
2. It enters the tail index as part of the product $$\alpha\delta$$. A dataset with $$\alpha\delta \approx 1.3$$ (barely finite mean, very heavy tail) combined with $$\delta \approx 1.8$$ (for a sensible mode) implies $$\alpha \approx 0.72$$. These are not free choices — they are forced by the parametrisation.

If the data wants a clear mode at moderate loss sizes *and* an extremely heavy extreme tail from catastrophic claims, Burr XII has only one knob to turn and it moves both controls simultaneously.

## The PowerBurr Fix

Liu & Meng add a fourth parameter τ > 0 acting on the exponent of the survival function:

$$S_\text{PB}(x;\, \alpha, \delta, \beta, \tau) = \left(1 + \left(\frac{x}{\beta}\right)^\delta\right)^{-\alpha\tau}$$

The CDF has closed form, so does the quantile function:

$$F_\text{PB}^{-1}(p) = \beta \cdot \left[(1-p)^{-1/(\alpha\tau)} - 1\right]^{1/\delta}$$

This is directly useful: VaR at any percentile is analytic. No numerical inversion for the 99.5th percentile SCR calculation. No quadrature for the limited expected value in XL layer pricing.

The tail index is now $$\alpha\delta\tau$$. Setting τ < 1 gives heavier-than-Burr-XII tails with the same (α, δ); τ > 1 gives lighter tails; τ = 1 recovers Burr XII exactly. Crucially, δ now controls the body shape without being forced to also determine how fat the extreme tail is. The coupling is broken.

## This Is GB2 — Actuaries Should Know This Name

The Generalised Beta of the Second Kind (GB2) is the four-parameter family that most severity distributions are special cases of:

```
GB2(μ, σ, ν, τ)
 ├── ν = 1  →  Burr XII
 │    ├── δ = 1  →  Pareto / Lomax
 │    └── α → ∞  →  Weibull
 └── PowerBurr  =  GB2 with ν free
```

Venter (1983) and McDonald (1984) formalised GB2 in the actuarial literature. PowerBurr is Burr XII with the ν = 1 constraint relaxed — not a new family, but a re-parametrisation from the Burr XII direction that makes the extra flexibility legible to practitioners who already work with Burr XII.

If you want to explore this now, without waiting for a dedicated implementation, the `gamlss` R package fits the full GB2 with covariate structure. Hold ν global and link τ to covariates:

```r
library(gamlss)
m <- gamlss(loss ~ vehicle_type + claim_type,
            nu.formula  = ~ 1,
            tau.formula = ~ claim_type + claimant_age,
            family = GB2,
            data = motor_bi_large_loss)
```

This is Liu & Meng's tail power regression, available today.

## The Regression — What Actually Matters for Pricing

The paper's contribution is not the distribution. It is the regression specification. Liu & Meng attach log-links to both the scale parameter and the tail power parameter:

$$\log(\beta_i) = \mathbf{x}_i^\top \mathbf{w}_\beta \qquad \log(\tau_i) = \mathbf{x}_i^\top \mathbf{w}_\tau$$

Shape parameters α and δ are held global across all observations to avoid identifiability problems. The regression coefficients **w**_β and **w**_τ are estimated jointly by L-BFGS-B with analytic gradients.

The interpretation is different from standard severity regression. **w**_β gives the scale relativities: vehicle type, geography, excess level affect mean loss size. **w**_τ gives something distinct: which risk characteristics predict that, conditional on a loss being large, it will be *catastrophically* large.

In motor BI, soft-tissue whiplash and catastrophic spinal injury can have similar initial reserves. Only one develops into a £10m periodic payment order. Injury type and claim status are plausible τ-covariates — they predict the shape of the extreme tail, not just the expected loss. This is a pricing question that standard scale-only severity regression cannot ask.

Our [insurance-severity](https://github.com/insurance-severity) library (`BurrTail`, `CompositeSeverityRegressor`) allows log-link covariates on scale β. It does not allow covariate-driven tail shape. PowerBurr regression addresses that gap directly. We do not have a `PowerBurrRegressor` implementation yet — more on that below.

## Where UK Pricing Teams Should Care

**Motor bodily injury.** The clearest application. The UK-specific driver is Ogden discount rates: at -0.25%, catastrophic injury awards are capitalised at very high multiples, producing losses in the £2m–£50m range. UK BI large-loss datasets including PPO claims typically show Hill estimator values for the tail index around 1.2–1.8 — close enough to the mean-existence boundary that model choice matters. A single Burr XII cannot simultaneously fit the soft-tissue mode and the PPO tail. PowerBurr can.

**Employers liability large loss.** Structurally identical. EL reinsurers pricing aggregate XL covers see only the extreme tail; the body of the EL claims distribution is often below their data attachment point. The coupling problem, if anything, is harder to diagnose because the body data that would help separate δ from the tail is not visible.

**Subsidence.** The ABI reported £153m paid in H1 2025 for approximately 9,000 subsidence claims, mean around £17k. The distribution is heavily right-skewed: cosmetic repair claims at £2k–£20k dominate by count, but full underpinning on clay-soil properties in south-east England runs £100k–£300k. UKCP18 projections for increasing summer dry spells suggest the extreme tail will thicken over the 2030s. PowerBurr's τ regression would allow soil type, property age, and tree proximity to enter the tail equation separately from the scale equation — a principled way to represent climate-driven tail drift.

**Where not to bother.** Lines with moderate, bounded severity — standard household contents, commercial property without large subsidence exposure — are fine with Burr XII or gamma. Flood event losses are driven by event-level correlation rather than individual severity; GPD/GEV at aggregate level is the right tool. Very extreme quantiles (99.9th percentile and above, reinsurance of reinsurers) still favour GPD because it is asymptotically justified by Pickands-Balkema-de Haan in a way PowerBurr is not.

## Limitations: Be Honest About These

PowerBurr has identifiability issues that Burr XII does not. The parameters α and τ appear in logpdf only through their product α·τ and through α·δ·τ. Separating them requires data that informs both body and tail — which whole-distribution fitting provides, but which is exactly what you lack if you are fitting to a claims data extract with a high reporting threshold.

The regression identifiability is trickier still. If the same features predict both scale and tail power — likely, since injury severity affects both average loss size and extreme tail weight — the separate **w**_β and **w**_τ coefficients may be correlated. Regularisation on **w**_τ is prudent.

There is no open code from Liu & Meng as of April 2026, and the paper is behind the ScienceDirect paywall (IME Vol. 127). The SSRN preprint (5148308) is also inaccessible. The formulations in this post are reconstructed from the SSRN abstract, Meng's prior GLMGA work (Li, Beirlant & Meng, ASTIN Bulletin 2021), and the GB2 family relationships. We believe they are correct but cannot verify the authors' exact parametrisation.

## What We Are Waiting For

Adding `PowerBurrDist` and `PowerBurrRegressor` to insurance-severity is roughly nine days of work. We are not doing it yet. Our trigger conditions: a practitioner request backed by a real UK dataset where Burr XII demonstrably fails on the tail, or a GIRO or CAS citation that signals the paper is gaining traction among working actuaries.

The gamlss route above is available now. If you run it on a motor BI large-loss dataset and find that τ regression meaningfully improves the tail fit — and that different claim types show materially different τ coefficients — send us the results. That is the evidence that would move this from interesting to implementable.

Run the Burr XII QQ plot first. If the top 1% sits above the line and you cannot fix it without breaking the body, PowerBurr is the next thing to try.

---

*Liu & Meng, "PowerBurr Regression Model for Heavy-Tailed Loss Data and its Application", Insurance: Mathematics and Economics Vol. 127 (March 2026). SSRN preprint 5148308 (February 2025). Mathematical formulations reconstructed from the SSRN abstract and GB2 family literature; full paper was not accessible during preparation of this post.*

*insurance-severity covers Burr XII (`BurrTail`), GPD tail, and spliced composites. PowerBurr regression is on the roadmap, not yet implemented.*
