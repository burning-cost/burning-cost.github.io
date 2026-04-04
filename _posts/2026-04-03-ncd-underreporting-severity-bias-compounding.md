---
layout: post
title: "NCD Underreporting Has a Second Problem You Are Probably Missing"
date: 2026-04-03
categories: [pricing, techniques]
tags: [ncd, bonus-malus, bms, underreporting, severity-bias, truncation, hunger-for-bonus, uk-motor, frequency-severity, glm, motor-pricing, cross-subsidy, arXiv-2601.12655, lemaire, claim-suppression, severity-distribution, combined-ratio, personal-lines]
description: "The hunger-for-bonus effect biases your NCD frequency relativities. It also biases your severity model. The two errors partially offset each other — but the combined underpricing is still around 13% at mid-ladder NCD classes, and the severity contamination means removing NCD from your severity model entirely is the correct fix."
math: true
author: burning-cost
---

We have written about NCD underreporting before. The short version: policyholders with good NCD rationally suppress small claims, your GLM is fit on the truncated data, and the resulting frequency relativities understate true risk at mid-ladder NCD classes by somewhere between 25% and 40%. The January 2026 formalisation by Liang et al. (arXiv:2601.12655) puts this on rigorous game-theoretic footing.

That is the frequency side of the problem. There is a severity side too, and it is almost never discussed.

---

## The truncation geometry

When a policyholder in NCD class $n$ suppresses all claims below threshold $b^*_n$, two things happen to your data simultaneously.

The first is the frequency effect you already know about. Observed frequency $\lambda^{\text{obs}}_n = \lambda_n \cdot P(Y > b^*_n)$. At 5-year NCD with $b^*_5 \approx £280$ and a Gamma severity distribution with mean ~£450 (broadly calibrated to a UK own-damage gross claim, before the policy excess is netted off), $P(Y > £280) \approx 0.65$. Observed frequency runs at 65% of true.

The second is a severity effect. The claims that do get reported in NCD class $n$ are not a random sample of the loss distribution — they are the claims that cleared the threshold. Your observed severity distribution at NCD class $n$ is **left-truncated at $b^*_n$**, not the full unconditional distribution.

If the true loss distribution is Gamma$(\alpha, \beta)$ with mean $\mu = \alpha/\beta$, then the observed conditional distribution is:

$$Y \mid Y > b^*_n \sim \text{Gamma}(\alpha, \beta) \text{ truncated from below at } b^*_n$$

The conditional mean of a left-truncated Gamma is strictly greater than $\mu$. For Gamma(2, 0.0044) — broadly calibrated to a UK own-damage gross severity of around £450 mean — the conditional mean given $Y > £280$ is approximately £609 rather than £450. That is a 34% upward distortion in observed mean severity at 5-year NCD relative to what a severity GLM fitted to the full (unsuppressed) data would give.

At 3-year NCD, where $b^*_3 \approx £370$, the conditional mean is approximately £683 — a 51% upward distortion.

At 9-year NCD, where $b^*_9 \approx £90$, the conditional mean is approximately £480 — a 6% distortion, essentially negligible.

---

## Why this compounds the frequency error

Your combined ratio at NCD class $n$ is (approximately) a function of:

$$\text{Loss ratio}_n \approx \frac{\lambda_n \cdot \mathbb{E}[Y_n]}{\text{premium}_n}$$

The standard GLM-based pricing workflow produces:
- A frequency relativity $r^{\text{freq}}_n$ fit on $\lambda^{\text{obs}}_n = \lambda_n \cdot p_n$ — understated by factor $p_n$
- A severity mean $\hat{\mu}^{\text{obs}}_n = \mathbb{E}[Y \mid Y > b^*_n]$ — overstated relative to the unconditional mean

Both models see the suppressed data. In a frequency-severity framework, the pure premium is $r^{\text{freq}}_n \cdot \hat{\mu}^{\text{obs}}_n$. The two errors run in opposite directions: frequency is understated (too few claims), severity is overstated (the claims that are reported are larger than average because the small ones were suppressed). They partially offset when multiplied together.

Work through the numbers for a 5-year NCD class on a £1,000 base, with a true expected loss rate (frequency × unconditional mean severity) of £100:

| Component | True value | Observed (suppressed) | Error |
|-----------|------------|----------------------|-------|
| Frequency $\lambda$ | 0.2203 claims/yr | 0.1433 claims/yr | −35% |
| Mean severity $\mu$ | £454 | £609 | +34% |
| Pure premium | £100 | £87.21 | −**13%** |

Frequency underestimated by 35%, severity overestimated by 34%. The product is $0.65 \times 1.34 = 0.87$. The pure premium is 13% below true — not 35% as the frequency-only calculation suggests. The severity bias substantially mitigates the combined error.

The severity partial offset depends on the distribution shape. For a heavy-tailed severity distribution (high coefficient of variation), the conditional mean given $Y > b^*_n$ rises sharply with $b^*_n$ — the offset is larger. For a lighter-tailed distribution, the offset is smaller. The offset is always less than the frequency error at reasonable UK severity distributions, so the combined error remains a systematic underpricing of mid-ladder NCD classes.

---

## The effect on your severity GLM

The severity bias creates a second, less obvious problem: it contaminates your severity GLM's NCD coefficient.

Most UK motor pricing teams include NCD as a factor in the severity model, or at minimum check for severity interaction effects by NCD band. The question the severity GLM asks is: does the expected loss amount differ by NCD class?

Under suppression, it does — but not because higher-NCD policyholders actually have larger accidents. It is because their observed claims are left-truncated at a higher threshold. A severity model fit on observed data will find a positive coefficient on high-NCD classes (higher observed mean severity) and attribute it to something about those customers — perhaps vehicle type, claim type mix, or some latent risk characteristic. The real explanation is the truncation geometry.

If you then use this severity coefficient to load the pure premium, you are partially correcting for the frequency underestimate but via the wrong mechanism. The frequency model sees lower claim rates and prices down; the severity model sees higher claim sizes and prices up; the net may look approximately right on average but is misspecified in its structure.

This matters for two reasons. First, if you change the NCD ladder — introducing a new class, changing step-back rules, modifying the discount schedule — the truncation thresholds shift, the severity GLM coefficient changes, and a model estimated under the old structure will misprice the new one. Second, if you introduce protected NCD as a product feature for some customers, their truncation threshold approaches zero, their observed severity distribution reverts to unconditional, and the severity coefficient will look different for the PNCD subgroup than for the unprotected group. A single NCD severity factor pools two structurally different truncation regimes.

---

## What you should actually see in the data

The severity bias is testable. Three empirical predictions follow directly from the truncation model:

**Prediction 1: The lower tail of observed severity should be more depleted at mid-ladder NCD classes.** Plot the empirical CDF of settled claim amounts separately for each NCD band. At 0-year NCD, the distribution should start near the policy excess and rise smoothly. At 5-year NCD, you should see a visible kink or thinning at the £200–400 range — the claims below the suppression threshold are absent, and the distribution should look as if it has been spliced from a higher starting point. If your claims data is sufficiently granular, this is visible with a few hundred claims per NCD band.

**Prediction 2: FNOL withdrawal rates should vary with NCD class.** A policyholder who calls to report a claim and then declines to proceed has almost certainly self-settled below the retention threshold. The FNOL withdrawal rate — calls opened but not converted to a paid claim — should be higher at mid-ladder NCD classes (3–6 years) than at 0-year NCD, and should follow roughly the inverted-U shape of the theoretical retention thresholds. If your FNOL system records the indicative loss amount at first notification, the size distribution of withdrawals by NCD class directly estimates $b^*_n$ without any model assumption.

**Prediction 3: Reported mean severity should rise then fall across the NCD ladder.** Suppression is an inverted-U with mid-ladder NCD having the highest thresholds (£280–370) and both ends lower (£90 at 9-year NCD, near-zero at 0-year). The conditional mean severity should therefore be highest at 3–5 years NCD and taper at both extremes. If your observed mean severity profile by NCD band shows this pattern — and it should — this is not risk heterogeneity; it is truncation geometry.

None of these predictions are exotic. Any pricing team with a claims ledger at policy level should be able to run tests 1 and 3 in an afternoon. Test 2 requires FNOL data with NCD class at incident date, which is more typically held in claims systems than policy systems and may need an extract from the claims team.

---

## The Norberg credibility complication

The severity truncation also corrupts Norberg-optimal BMS relativities, though this point is typically only relevant if you are calibrating the NCD scale explicitly from first principles rather than fitting it as a GLM factor.

Norberg's (1976) credibility criterion for the BMS relativity at class $n$ is the asymptotic minimum-MSE estimator of the long-run expected claim cost for a policyholder at steady state in class $n$. Under a Poisson-Gamma heterogeneity model, this is the Bühlmann credibility posterior mean of the individual's claim intensity given their NCD history.

If the claims data used to estimate the Bühlmann parameters are drawn from the suppressed distribution — which they are, for the same reasons the GLM is biased — the posterior mean will underestimate the claim intensity at high-NCD classes. The between-class variance parameter $a$ in the Bühlmann formula is estimated from the variation in observed (suppressed) claim counts across policyholders, which will understate the true heterogeneity at high-NCD classes. The resulting credibility factors will be too small, concentrating the relativity scale toward the mean and compressing the true differentiation between risk bands.

The direction of the Norberg bias is the same as the GLM bias — it underestimates the risk premium for high-NCD policyholders — but the magnitude differs because credibility estimation weights by exposure, and the suppression rate varies across the NCD distribution. Teams using explicit Bühlmann-Straub credibility for NCD calibration (rather than a straight GLM) face the same correction requirement but through different algebra.

---

## The correction for the severity model

If you implement Lemaire's algorithm for your NCD ladder and compute the theoretical retention thresholds $\{b^*_n\}$, the frequency correction is:

$$\hat{\lambda}_n = \lambda^{\text{obs}}_n / p_n, \quad p_n = 1 - F_{\text{Gamma}}(b^*_n; \hat{\alpha}, \hat{\beta})$$

The severity correction requires a different calculation. The observed conditional mean at NCD class $n$ is:

$$\mu^{\text{obs}}_n = \mathbb{E}[Y \mid Y > b^*_n] = \frac{\int_{b^*_n}^{\infty} y f(y;\hat{\alpha},\hat{\beta})\, dy}{P(Y > b^*_n)}$$

For a Gamma distribution, this has a closed form involving the regularised incomplete gamma function. The true unconditional mean is $\hat{\mu} = \hat{\alpha}/\hat{\beta}$, which is insensitive to the truncation (since it does not depend on $n$). The corrected severity relativity at class $n$ is therefore:

$$\text{severity correction}_n = \hat{\mu} / \mu^{\text{obs}}_n = \frac{\hat{\alpha}/\hat{\beta}}{\mathbb{E}[Y \mid Y > b^*_n]}$$

This is a deflation factor less than 1 for all NCD classes with positive suppression — the correction lowers the severity relativity because the observed mean is inflated by truncation.

In a frequency-severity framework, the pure premium correction is the product of both corrections:

$$\hat{\lambda}_n^{\text{true}} \cdot \hat{\mu}^{\text{true}} = \frac{\lambda^{\text{obs}}_n}{p_n} \cdot \frac{\hat{\alpha}/\hat{\beta}}{\mathbb{E}[Y \mid Y > b^*_n]} = \lambda^{\text{obs}}_n \cdot \hat{\mu}$$

The result is clean: the corrected pure premium at each NCD class equals the observed claim count divided by $p_n$, multiplied by the *unconditional* mean severity $\hat{\mu}$ — not the conditional mean you would get from a naive severity GLM. The pure premium correction collapses entirely into the frequency correction, and the true severity is the same across all NCD classes (since the loss distribution is assumed homogeneous across classes in the model).

In practice, loss distributions are not homogeneous across NCD classes — larger vehicles have different severity distributions from smaller ones, own damage and third-party property have different distributions, and the mix varies by NCD band. The correction needs to be applied separately by cover type and vehicle segment, using the appropriate severity parameters for each. A single portfolio-level correction will be accurate on average but wrong for specific cells.

---

## A note on separability

This analysis assumes suppression is the dominant explanation for severity variation by NCD class. There are two confounds worth acknowledging.

First, genuine adverse selection: higher-NCD drivers may, on average, use cheaper vehicles or drive in less congested areas, producing genuinely lower claim amounts independent of suppression. This would create a genuine negative severity-NCD relationship, partially offsetting the positive suppression effect. Whether it dominates depends on your portfolio. The truncation-induced effect should dominate at mid-ladder NCD classes (3–6 years, where thresholds are highest), and the genuine risk heterogeneity is the more plausible explanation at the extremes.

Second, claim type mix: own damage claims are fully suppressible — the policyholder decides whether to claim for their own car. Third-party claims are partially non-suppressible — if you hit someone else's vehicle, they may claim against you regardless. The truncation effect operates primarily on own-damage frequency. A portfolio with a high proportion of third-party-only policies will have a smaller suppression effect and a smaller severity truncation distortion.

The clean version of the correction applies separately to own-damage and third-party claim types, using type-specific severity distributions and the appropriate suppression threshold. Own damage with its fully suppressible nature should show the full truncation pattern. Third party should show a muted version.

---

## What this changes about the four-step correction

In our [earlier piece on correcting NCD relativities](/pricing/motor/techniques/2026/04/02/your-ncd-relativities-are-wrong/), we outlined a four-step correction: estimate retention thresholds, compute censoring probability per class, re-estimate the frequency model, iterate. That corrects the frequency model.

The severity correction adds a fifth consideration: do not use your NCD-segmented severity model to set severity relativities by NCD class. The NCD coefficient in your severity GLM is contaminated by truncation geometry. The corrected severity at each NCD class, for a homogeneous loss distribution, is the same unconditional mean regardless of NCD band. NCD should not appear as a factor in your severity model at all — the apparent variation is a data artefact, not a genuine risk signal.

If you are running frequency and severity models independently and multiplying out, removing NCD from the severity model and pushing the full correction into the frequency model is the cleaner specification. If you are running a Tweedie model, the Tweedie parameter interaction between frequency and severity contamination is more complex, and a clean correction requires explicitly modelling the truncated count-and-size likelihood at each NCD class rather than using a single Tweedie.

---

## The bottom line

The hunger-for-bonus frequency bias is roughly 13% underpricing at the pure premium level for a 5-year NCD class, once the severity partial offset is accounted for. This is smaller than the raw frequency error of 35%, because the severity model overstates claim amounts for the same reason the frequency model understates claim counts — the truncation geometry inflates the observed severity mean by approximately 34% at 5-year NCD.

The combined error is systematic and class-specific. It peaks at mid-ladder NCD (3–6 years), is smaller at the top of the ladder (9+ years), and approaches zero at the bottom. Applying a flat combined-ratio loading uniformly across NCD classes will miss the pattern. The correction needs to be class-by-class, cover-type-specific, and derived from your actual NCD ladder structure and severity distribution.

That the two errors substantially offset is not a reason to ignore either. The partial offset does not make the underpricing acceptable — it reduces a 35% frequency error to a 13% combined error. But the structural misspecification of the severity model — attributing truncation geometry to genuine risk heterogeneity — remains a problem regardless of the net pricing error. The severity GLM is learning the wrong thing, and that has consequences when the NCD ladder changes.

---

## Further reading

- [Your NCD Relativities Are Wrong, and the Maths Now Tells You How Wrong](/pricing/motor/techniques/2026/04/02/your-ncd-relativities-are-wrong/) — the frequency correction in detail
- [The Hunger for Bonus: How UK Motor NCD Pricing Gets the Frequency Wrong](/pricing/techniques/2026/04/01/the-hunger-for-bonus-how-uk-motor-ncd-pricing-gets-the-frequency-wrong/) — the full theoretical treatment with the Liang et al. paper
- Liang, Z., Zhang, J., Zhou, Z., Zou, B. (2026). "Optimal Underreporting and Competitive Equilibrium." [arXiv:2601.12655](https://arxiv.org/abs/2601.12655)
- Lemaire, J. (1977). "La Soif du Bonus." *ASTIN Bulletin* 9(1-2):181–190
- Norberg, R. (1976). "A credibility theory for automobile bonus systems." *Scandinavian Actuarial Journal*, 2:92–107
