---
layout: post
title: "When Burr XII Isn't Fat Enough — The PowerBurr Family for Large-Loss Severity Modelling"
date: 2026-04-02
categories: [insurance-pricing, severity]
tags: [severity-modelling, burr-xii, gb2, powerburr, heavy-tails, tail-index, motor-insurance, bodily-injury, xl-reinsurance, insurance-severity, Liu-Meng-2026, regression, spliced-distributions, actuarial]
description: "Burr XII's body and tail are controlled by the same parameters — you can't fix one without breaking the other. Liu & Meng's PowerBurr adds a fourth parameter that decouples them. Here is what it actually does and where it earns its keep on UK large-loss books."
math: true
author: burning-cost
---

You've fitted every distribution in your toolkit to the motor BI large-loss dataset. Log-normal is out on AIC by a country mile — the tail is far too light. Weibull fails similarly. Burr XII wins, comfortably, with α̂ = 1.4, δ̂ = 1.8, β̂ = 12,000. The QQ plot looks reasonable up to the 97th percentile. Then the top twenty claims — the PPO awards, the catastrophic spinal injuries priced under Ogden tables with a -0.25% discount rate — all sit above the 45-degree line.

You try reducing α to fatten the tail. The AIC improves slightly at the extreme, but now the body fit degrades: the density peaks too early, the 50th-to-90th percentile range is wrong, and your XL layer costs for the £100k xs £50k layer have shifted by 8%. You've traded one problem for another.

This is the parameter coupling trap of Burr XII, and it is the problem Liu & Meng address in their PowerBurr regression model, published in *Insurance: Mathematics and Economics* Vol. 127 (March 2026).

---

## The Coupling Problem in Burr XII

The Burr XII (Singh-Maddala) distribution has three parameters and the following survival function:

$$S(x;\, \alpha, \delta, \beta) = \left(1 + \left(\frac{x}{\beta}\right)^\delta\right)^{-\alpha}$$

The tail behaves as $$S(x) \sim C \cdot x^{-\alpha\delta}$$ as $$x \to \infty$$. The tail index is therefore $$\alpha\delta$$.

The second parameter δ does two jobs simultaneously. It controls the body shape: the mode of the distribution (when it exists) is:

$$\text{mode} = \beta \cdot \left(\frac{\delta - 1}{\alpha\delta + 1}\right)^{1/\delta}, \quad \delta > 1$$

And it enters the tail index as part of the product α·δ. If your data has a sharp, well-defined mode at around £15k and also has an extreme tail with finite but barely-positive mean (α·δ close to 1.2), you have two constraints and only two free parameters after fixing scale. The body shape and the tail heaviness are not independently negotiable. You are stuck.

The severity modeller's usual response is to use a spliced model — fit a light-tailed body distribution and a GPD tail above some threshold. This is the approach taken by the `LognormalGPD` and `GammaGPD` composites in our [insurance-severity](https://github.com/insurance-severity) library, and for good reason: it works, and the GPD tail has theoretical backing from the Pickands-Balkema-de Haan theorem. But it introduces a different headache: threshold selection. The choice of splice point materially affects layer costs, and in practice modellers spend considerable time justifying that choice.

---

## What the PowerBurr Does

Liu & Meng introduce a fourth parameter τ > 0 acting directly on the survival function:

$$S_\text{PB}(x;\, \alpha, \delta, \beta, \tau) = \left(1 + \left(\frac{x}{\beta}\right)^\delta\right)^{-\alpha\tau}$$

The corresponding density is:

$$f_\text{PB}(x;\, \alpha, \delta, \beta, \tau) = \alpha\tau\delta\,\beta^{-\delta}\,x^{\delta-1} \cdot \left(1 + \left(\frac{x}{\beta}\right)^\delta\right)^{-(\alpha\tau+1)}$$

This is structurally identical to Burr XII with shape parameter α* = α·τ. The CDF has closed form:

$$F_\text{PB}(x) = 1 - \left(1 + \left(\frac{x}{\beta}\right)^\delta\right)^{-\alpha\tau}$$

And so does the quantile function:

$$F_\text{PB}^{-1}(p) = \beta \cdot \left[(1-p)^{-1/(\alpha\tau)} - 1\right]^{1/\delta}$$

The tail index is now α·δ·τ. Setting τ = 1 recovers standard Burr XII; τ < 1 gives a heavier tail for the same (α, δ); τ > 1 gives a lighter tail.

Crucially, δ now controls the body shape alone. If you want the mode at £15k and a heavy tail with index 1.1, you can achieve both by setting δ appropriately for the mode and τ < 1 to bring the tail index down, without the two objectives fighting each other.

---

## This is GB2 Viewed from the Burr XII Direction

The Generalised Beta of the Second Kind (GB2) is the four-parameter family that encompasses most of the distributions actuaries actually use for severity:

```
GB2(μ, σ, ν, τ)
 │
 ├── ν = 1            → Burr XII (Singh-Maddala)
 │    ├── δ = 1       → Pareto / Lomax
 │    └── α → ∞      → Weibull (limiting case)
 │
 ├── τ = 1            → Generalised Gamma
 ├── σ = 1            → Pearson VI / Beta Prime
 └── PowerBurr        = GB2 with ν free (or equivalently, Burr XII × τ)
```

GB2 was formalised in the actuarial literature by Venter (1983) and McDonald (1984). Actuaries who know their distribution families will immediately recognise PowerBurr as Burr XII with the ν = 1 constraint removed — not a fundamentally new invention, but a re-parametrisation that makes the extension legible from the Burr XII direction.

This matters practically: the `gamlss` R package already fits GB2 with full covariate structure via `GB2()`. If you want to explore PowerBurr regression today, before any dedicated implementation exists, this is your route:

```r
library(gamlss)
m <- gamlss(loss ~ vehicle_type + claim_type,
            sigma.formula  = ~ 1,
            nu.formula     = ~ 1,        # hold nu global ≈ Burr XII body
            tau.formula    = ~ claim_type + claimant_age,
            family = GB2,
            data = motor_bi_large_loss)
```

The `tau.formula` here is driving what Liu & Meng call the tail power regression.

---

## The Regression Model — And Why It Matters for Pricing

The distribution alone is a minor incremental improvement. The genuine contribution in Liu & Meng (2026) is the regression specification:

$$\log(\beta_i) = \mathbf{x}_i^\top \mathbf{w}_\beta$$

$$\log(\tau_i) = \mathbf{x}_i^\top \mathbf{w}_\tau$$

Global (shared across all observations) shape parameters α and δ are estimated jointly with the regression coefficients **w**_β and **w**_τ. The log-likelihood is:

$$\ell = \sum_i \left[\log\alpha + \log\tau_i + \log\delta - \log\beta_i + (\delta-1)\log(y_i/\beta_i) - (\alpha\tau_i + 1)\log\left(1 + (y_i/\beta_i)^\delta\right)\right]$$

Analytic gradients with respect to **w**_τ are available; L-BFGS-B handles the optimisation. Initialising from Burr XII MLE (τ = 1) gives stable convergence.

What this means for pricing: **w**_β contains the standard scale relativities — vehicle age drives mean loss size, geography drives frequency, and so on. **w**_τ contains something different: the relativities that predict whether, conditional on a loss being large, it will be *catastrophically* large. These are not the same features.

In motor BI, injury type and claim status are natural candidates for **w**_τ. A soft-tissue whiplash claim and a catastrophic spinal injury might have similar initial reserve sizes. Only one of them will develop into a £10m PPO. The τ regression is trying to model that distinction — not just "how big will this claim be" but "how fat is the extreme-loss distribution conditional on this claim's characteristics."

No pricing model we are aware of, outside of ad hoc GPD threshold adjustments, currently represents this. Our `BurrTail` class and `CompositeSeverityRegressor` in insurance-severity allow log-link covariates on the scale β; they do not allow covariate-driven tail shape. PowerBurr regression addresses that gap directly.

---

## UK Applications — Where the Coupling Problem Actually Bites

Not every line of business needs this. Here is where PowerBurr earns its extra parameter:

**Motor bodily injury.** The combination of a well-defined mode (bulk soft-tissue claims at £10k–£20k) and an extremely heavy extreme tail (PPO awards for catastrophic injuries at £2m–£50m under Ogden rates) is precisely the scenario where Burr XII's α·δ coupling is most painful. UK datasets including PPO claims typically produce Hill estimator values for the tail index around 1.2–1.8, which is dangerously close to variance-infinite territory. A single Burr XII cannot simultaneously hit the mode and the PPO tail. PowerBurr can.

**Employers liability large loss.** Structurally identical to motor BI. EL reinsurers pricing aggregate XL covers see only the extreme tail of the underlying distribution; the coupling problem is if anything worse because the body of the EL claims distribution is not visible in their data.

**Subsidence.** The ABI reported £153m paid in H1 2025 for subsidence, across approximately 9,000 claims at an average of around £17k. But the distribution is heavily skewed: the bulk of claims are £2k–£20k (cosmetic repairs, drainage investigations), while full underpinning on clay-soil properties in southeast England can reach £100k–£300k. The UKCP18 projections for increasing summer dry spells in SE England suggest this tail will get heavier over the coming decades. PowerBurr's τ regression provides a principled way to model soil type, property age, and proximity to mature trees as tail-shape covariates distinct from scale covariates.

**Where PowerBurr adds little.** Lines with moderate, well-behaved severity (household contents, standard commercial property away from subsidence risk) are adequately handled by Burr XII or even gamma. The four-parameter overhead is not justified. Flood event losses are dominated by event-level correlation rather than individual severity; GPD/GEV at the aggregate level is the right tool. High-frequency lines where the GPD body approximation is already stable are fine as-is.

---

## What We Don't Have Yet

The paper is behind the ScienceDirect paywall (IME Vol. 127, March 2026). The SSRN preprint (5148308) is also inaccessible at the time of writing. The mathematical formulations in this post are reconstructed from the SSRN abstract, Meng's prior GLMGA work (Li, Beirlant & Meng, ASTIN Bulletin 2021, arXiv:1912.09560), and the GB2 family relationships — we believe these are correct but cannot verify the authors' exact parametrisation or the empirical AIC improvements they report.

There is no open code from Liu & Meng. As of April 2026, no GitHub repository is visible. The paper uses catastrophe loss data from China (consistent with the UIBE affiliation); UK actuaries should treat tail index estimates from Chinese earthquake and flood losses with appropriate scepticism — the same distributional family may fit, but the calibrated parameters will differ.

Our insurance-severity library does not yet include a `PowerBurrDistribution` class or a `PowerBurrRegressor`. Adding both would be roughly nine days of work (distribution, MLE, regression, tests, documentation). We are watching for either a practitioner request against a real dataset, or a UK actuarial conference citation, before committing to a PR. If the gamlss route above surfaces a strong empirical result on a UK motor BI dataset, that would be the trigger.

---

## The Honest Assessment

PowerBurr is not a revolution. It is Burr XII with the strait-jacket loosened. The tail-body coupling in Burr XII is a genuine constraint that practitioners encounter on specific datasets, and a fourth parameter targeting exactly that coupling is more useful than the quarterly stream of "exponentiated-Kumaraswamy-Burr-XII-Harris-G" constructions that appear in the actuarial distribution literature without ever making contact with a real pricing problem.

What makes Liu & Meng's work worth reading is the regression formulation. Separating scale relativities from tail-power relativities — letting claim type and injury severity enter the tail equation independently of how they enter the scale equation — is a pricing idea we have not seen operationalised this cleanly before. That, not the distribution itself, is what we would want to implement.

Whether the extra complexity is justified depends on whether your data actually shows the coupling problem. Run the Burr XII QQ plot first. If the top 1% sits above the line and you cannot fix it without breaking the body, PowerBurr is the next thing to try.

---

*PowerBurr distribution: Liu & Meng, "PowerBurr Regression Model for Heavy-Tailed Loss Data and its Application", Insurance: Mathematics and Economics Vol. 127 (2026). SSRN preprint 5148308 (February 2025).*

*The mathematical formulations above are consistent with the SSRN abstract and the GB2 family literature; the full paper was not accessible during preparation of this post.*

*Our [insurance-severity](https://github.com/insurance-severity) library covers Burr XII (`BurrTail`), GPD tail, and spliced composites. PowerBurr regression is on the roadmap but not yet implemented.*
