---
layout: post
title: "Two Ways to Control Risk in Automated Underwriting: Conditional vs Marginal"
date: 2026-04-01
categories: [techniques, underwriting]
tags: [conformal-prediction, selective-prediction, underwriting, STP, e-values, SCRC, SCoRE, MDR, SDR, FCA-Consumer-Duty, insurance-conformal, automated-underwriting, motor, risk-control, arXiv-2603-24704, arXiv-2512-12844, Bai-Jin, Xu-Guo-Wei, finite-sample-guarantee, python]
description: "Two rigorous frameworks for automated underwriting triage — SelectiveConformalRC (SCRC) controls expected loss on your auto-priced book; SCoRE controls total deployed risk via e-values. The difference is not academic: one maps to combined ratio governance, the other to sequential cost budgets. We explain both and show when each applies."
math: true
author: burning-cost
---

Your motor pricing GBM produces a quote for every incoming risk. For most risks — mainstream postcodes, typical claim history, occupation in the standard database — the model is operating well within its training distribution and the quote is reliable. For some risks it is not: unusual vehicle modifications, edge-case occupation codes, risks with sparse data in surrounding rating cells. For those, the model is extrapolating, and you may not want to auto-price them.

The question is: how do you decide which is which? "Uncertainty > threshold" is the usual answer. But what threshold? What does uncertainty mean here? And what guarantee does crossing it actually provide?

Two recent papers give rigorous answers to this question, and they give *different* answers — controlling different quantities, suited to different governance contexts. Both are now relevant to UK pricing teams facing FCA Consumer Duty obligations around automated decision-making.

---

## The underwriting triage problem

Straight-through processing (STP) pipelines make a binary decision per incoming risk: auto-price or refer to an underwriter. The auto-pricing path is faster, cheaper, and scales to volume personal lines. The referral path costs time and resource but catches risks where the model is unreliable.

The design question is how to set the boundary. The naive approach is to set it on a model confidence metric — max softmax probability, predicted vs observed variance, entropy. These are reasonable proxies but they do not provide formal guarantees about what happens to your book if you deploy at that boundary.

Formal guarantees require a framework that connects the deployment criterion to a property of the accepted book. Two frameworks now exist that do this rigorously, with finite-sample validity under exchangeability only.

---

## Framework 1: Conditional risk control (SCRC)

**What it guarantees:** $\mathbb{E}[L \mid \text{accepted}] \leq \alpha$

SCRC — Selective Conformal Risk Control, from Xu, Guo and Wei (arXiv:2512.12844, 2025) — finds a threshold $\lambda$ on a selection score $s(x)$ such that the conditional expected loss on the accepted book is at most $\alpha$.

In underwriting terms: if $L = \mathbf{1}\{\text{large claim}\}$, setting $\alpha = 0.08$ guarantees that the expected large-claim rate on your auto-priced book is at most 8%, with a finite-sample correction. Not approximately. Not asymptotically. For any sample size, given exchangeability of your calibration data.

The mechanism is a two-stage calibration. Stage one sets $\lambda_1$: the minimum selection score for acceptance, ensuring you auto-price at least a required fraction $\xi$ of incoming risks. Stage two sets $\lambda_2$: a secondary risk threshold, calibrated so the expected loss on the selected subset satisfies the budget constraint $k = \lceil (n+1) \cdot \alpha \cdot \hat{\xi}_{\text{LCB}} \rceil - 1$, where $\hat{\xi}_{\text{LCB}}$ is a DKW lower confidence bound on the selection rate.

This is fully implemented in [`insurance-conformal`](/insurance-conformal/) as `SelectiveConformalRC`:

```python
from insurance_conformal.risk import SelectiveConformalRC
from insurance_conformal.risk.selection_scores import selection_score_msp

# Build selection scores from model probabilities
scores_cal = selection_score_msp(model.predict_proba(X_cal))

# Build the loss matrix across a lambda_2 grid
# pred_sets_cal[j, i] = loss for obs i at threshold index j
scrc = SelectiveConformalRC(
    alpha=0.08,  # E[large claim | auto-priced] <= 8%
    xi=0.75,     # must accept at least 75% of risks
    method="SCRC-I",
)
scrc.calibrate(y_cal, scores_cal, pred_sets_cal)

# Apply to new risks
decisions = scrc.predict(scores_new)
# decisions["selected"]: True = auto-price, False = refer

print(scrc.underwriting_summary())
# {'selection_rate': 0.81, 'conditional_risk': 0.064, 'alpha': 0.08, ...}
```

The guarantee is on the auto-priced book. If a risk scores below $\lambda_1$, it is referred — the guarantee says nothing about what happens to referred risks, which is appropriate: those go to a human underwriter.

---

## Framework 2: Marginal deployment risk (SCoRE)

**What it guarantees:** $\mathbb{E}[L \cdot \psi] \leq \alpha$

SCoRE — Selective Prediction with General Risk Control, from Bai and Jin at Stanford (arXiv:2603.24704, March 2026) — controls a different quantity: the expected product of loss and deployment indicator across *all* arriving risks, including the ones you decline to auto-price.

The mechanism uses e-values rather than threshold calibration. For each incoming risk $x$, SCoRE computes a risk-adjusted e-value $E_\gamma(x) \geq 0$ from the calibration data. The deployment rule is: auto-price iff $E_\gamma(x) \geq 1/\alpha$.

The e-value is constructed so that $\mathbb{E}[L \cdot E_\gamma] \leq 1$ — a supermartingale condition that, when thresholded at $1/\alpha$, guarantees $\mathbb{E}[L \cdot \psi] \leq \alpha$ (Theorem 3.2 of Bai and Jin). This is a marginal guarantee: it holds over the full population of arriving risks, not just over the accepted subset.

There is also a batch variant — SCoRE-SDR — that controls the average loss per deployed prediction across a batch using e-Benjamini-Hochberg. For a batch of $m$ test points, compute one e-value per point, apply e-BH to find the largest deployable set $R$, and you get $\mathbb{E}[\text{avg loss on } R] \leq \alpha$. This is the FDR analogue for continuous losses.

The SCoRE `SCoRESelector` class is on our Phase 41 build queue (not yet in the library). The API will look like:

```python
# Proposed API — not yet built, queued for insurance-conformal v0.9.0
from insurance_conformal.risk import SCoRESelector

selector = SCoRESelector(alpha=0.08, method="MDR")
selector.calibrate(losses_cal=large_claim_flags, scores_cal=model_scores)

# Per-risk e-values for new batch
e_vals = selector.e_values(scores_new)
deploy = selector.select(scores_new)  # deploy iff e_value >= 1/alpha
```

---

## The critical difference: a numerical example

Both frameworks use the same $\alpha = 0.08$. On a portfolio of 100 auto-priced risks, suppose:

- 80 are accepted by both frameworks
- Those 80 have a large-claim rate of 0.09 (9%)

**SCRC check:** $\mathbb{E}[L \mid \text{accepted}] = 0.09 > 0.08$. *Guarantee violated.*

**SCoRE MDR check:** $\mathbb{E}[L \cdot \psi] = 0.09 \times 0.80 = 0.072 \leq 0.08$. *Guarantee satisfied.*

The MDR guarantee is satisfied because the 20% non-deployed risks dilute the joint expectation below $\alpha$. But the conditional expected loss on your actual auto-priced book is 9%, above the stated target.

This matters for UK actuarial reporting. The question a Chief Actuary or pricing committee asks is almost always: "what is the expected large-claim rate on our auto-priced book?" That is the conditional quantity, and it maps directly to combined ratio targets. SCoRE MDR answers a different question: "what is the total risk accumulated by the auto-pricing decision across all risks that arrived?" This is coherent and formally valid, but it is not what combined ratio governance is measuring.

We think SCRC is the right default for UK personal lines STP governance. SCoRE's MDR framing is better suited to contexts where you are tracking a rolling cost budget across multiple decision cycles — or where the natural governance unit is total deployed mispricing, not error rate on the accepted book.

---

## When would you actually use each?

**Use SCRC when:**

- Your governance question is "what is the expected loss rate on risks we auto-price?" — this is most combined ratio and Consumer Duty reporting
- You have a fixed underwriting model and want a calibrated acceptance threshold
- You need a real-time STP API: SCRC-I calibration is $O(n)$ amortised, fast enough for sub-100ms quote paths
- Your actuarial committee will review the output: `underwriting_summary()` and `pareto_report()` give interpretable diagnostics

**Use SCoRE when:**

- You are processing a bulk book transfer overnight — an MGA cedant submitting 50,000 new risks, a renewal wave, a portfolio audit. The SDR variant with e-BH is designed for batch deployment: process the whole book, find the maximum deployable subset with average loss $\leq \alpha$, output the accept/refer list
- Your governance frame is a sequential risk budget: "we will accumulate no more than $X$ total mispricing cost from automated decisions over the quarter." MDR tracks that budget; conditional risk does not
- You want the Neyman-Pearson optimality result from Theorem 4.6: the optimal selection score for MDR is $\ell(x)/r(x)$ — conditional expected loss divided by a reference measure. This gives you a principled way to construct better underwriting selection scores, not just to apply a threshold to whatever your GBM output happens to be

---

## FCA Consumer Duty: what these frameworks actually provide

FCA Consumer Duty (PS22/9, effective July 2023) creates three pressures on automated underwriting systems.

**Competence boundary (PRIN 12):** Firms must act in customers' interests. Deploying a pricing model on risks it is not competent to price is a governance failure — it can produce arbitrary prices for edge cases, which Consumer Duty's fair value obligation is precisely intended to prevent. Both SCRC and SCoRE provide an auditable, statistically principled criterion for when to withhold a model decision. This is substantially better governance than an opaque "refer if uncertainty score > 0.3" rule that nobody can explain to a regulator.

**Outcomes monitoring (PS22/9 para 11):** Firms must monitor outcomes of automated decisions. SCRC's conditional risk maps cleanly to loss ratio monitoring on the STP segment. SCoRE's SDR maps to average harm per deployed prediction — a natural metric for monitoring if you are running the e-BH procedure on renewal batches.

**Product oversight (PROD 4):** Pricing governance requires documented, reproducible decision criteria. An e-value threshold or a calibrated SCRC boundary provides an audit trail that a threshold on model entropy does not.

To be direct about what this is not: the FCA has not endorsed conformal prediction or e-values as governance tools. Consumer Duty is outcomes-based — the FCA cares whether customers are harmed, not whether you used a specific statistical method. The practical value of these frameworks is that they give you mathematically defensible referral criteria with clear properties, not regulatory safe harbour.

---

## Calibration data requirements

Both frameworks require exchangeable calibration data — claims that were priced by the same model on a similar population to your deployment context. In practice, this means using a recent hold-out set from your existing motor or home pricing GBM, not a historical book written under different rating factors.

A rough guide for SCRC at $\alpha = 0.08$, $\xi = 0.75$, $\delta = 0.10$:

| Calibration n | DKW correction (eps) | Budget k | Notes |
|---|---|---|---|
| 500 | 0.049 | 29 | Borderline: small k, risk of infeasibility |
| 1,000 | 0.035 | 59 | Workable for most motor segments |
| 2,000 | 0.025 | 119 | Recommended minimum for stable calibration |
| 5,000 | 0.015 | 299 | Comfortable; Pareto frontier is well-resolved |

For SCoRE MDR, the e-value computation per test point is $O(n^2)$ in the worst case (Algorithm 3 of Bai and Jin reduces the continuous infimum to a finite search over the calibration score set). At $n = 10{,}000$ calibration points, this is borderline for real-time APIs but workable for overnight batch jobs. For real-time STP, use SCRC.

---

## What is already built

`insurance-conformal` currently implements SCRC in full:

- `SelectiveRiskController`: simple SCRC-I with DKW correction, single-stage
- `SelectiveConformalRC`: full two-stage SCRC-I and SCRC-T with Pareto frontier, `underwriting_summary()`, `pareto_report()`
- `selection_scores.py`: MSP, margin, entropy, and energy score functions for constructing the selection score $s(x)$

The SCoRE e-value mechanism — `e_value_mdr()`, `e_value_sdr()`, `e_bh_procedure()`, `SCoRESelector` — is on the build queue for `insurance-conformal` v0.9.0. The mathematical machinery is non-trivial to implement correctly: the infimum over $\ell \in [0,1]$ in Equation 4.1 of Bai and Jin requires careful handling of near-zero denominators and score ties. We are prioritising correctness over speed of delivery.

---

## The practical recommendation

For a UK motor pricing team setting up or reviewing an STP pipeline today: use SCRC via `SelectiveConformalRC`. It controls the quantity your governance framework is actually measuring, it is fast enough for real-time deployment, and it is fully implemented with actuarial-facing diagnostics.

For MGA book processing, overnight renewal waves, or any setting where you are making batch accept/refer decisions and your governance metric is total deployed risk rather than error rate on the accepted book: SCoRE-SDR with e-BH will be the better tool once it is built.

The two approaches are not competing. They answer different questions. The mistake is assuming the question "should this risk be auto-priced?" has a single rigorous answer — it has at least two, depending on what you are trying to control.

---

## The papers

Xu, Rui, Han Guo, and Rina Foygel Barber. "Selective Conformal Risk Control." arXiv:2512.12844 [stat.ML]. December 2024.

Bai, Tian, and Ying Jin. "Conformal Selective Prediction with General Risk Control." arXiv:2603.24704 [stat.ML]. March 2026.

---

## Related posts

- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/techniques/pricing/uncertainty/2026/03/13/insurance-conformal-risk/) — conformal risk control and the `insurance-conformal` library overview
- [Conformal Prediction for Insurance Pricing: Intervals, Risk Control, and the Practical Toolkit](/techniques/pricing/2026/03/23/does-conformal-prediction-work-insurance-pricing/) — when conformal prediction works, when it does not, and what calibration data you actually need
- [Conditional Coverage in Conformal Prediction: Model Selection with CVI](/techniques/pricing/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — how to check whether your conformal guarantees hold conditionally, not just marginally
- [KMM Conformal Prediction under Covariate Shift in Insurance Pricing](/techniques/pricing/2026/03/31/kmm-conformal-prediction-covariate-shift-insurance-pricing/) — when your calibration and deployment distributions differ
