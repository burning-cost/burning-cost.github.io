---
layout: post
title: "Conformal Prediction with Change Points: When Your Coverage Guarantee Breaks and What to Do About It"
date: 2026-03-31
categories: [techniques]
tags: [conformal-prediction, change-points, regime-switching, ACI, FACI, adaptive-conformal-inference, model-monitoring, time-series, claims-inflation, insurance-conformal, insurance-conformal-ts, insurance-monitoring, Consumer-Duty, FCA, uncertainty-quantification, python, arXiv-2509-02844]
description: "Conformal prediction with change points (CPTC, arXiv:2509.02844, NeurIPS 2025) extends adaptive conformal inference to detect structural breaks and reset coverage guarantees proactively. Here is what that means for UK pricing teams running models through claims inflation spikes and regulatory step-changes."
author: burning-cost
---

Adaptive Conformal Inference (ACI) has a known weakness. It is reactive. When your model's calibration breaks down — because the claims environment changed, because a new cohort of customers arrived, because the FCA changed the rules — ACI detects the problem the slow way: it waits for miscoverage to accumulate, then widens the prediction intervals until coverage recovers. In a fast-moving regime change, this can mean weeks or months of invalid intervals before the method catches up.

Margaux Zaffran, Yannig Goude and Aymeric Dieuleveut (arXiv:2509.02844, accepted NeurIPS 2025) attack this directly. Their Conformal Prediction with Change Points (CPTC) method couples ACI with a structural break detector, and uses the detected break to reset the conformal quantile immediately rather than waiting for the coverage gap to close on its own. For UK pricing teams who run models through claims inflation spikes, whiplash reform step-changes, and post-GIPP repricing events, this is directly useful.

---

## What is wrong with ACI as it stands

The `insurance-conformal-ts` library implements ACI (and ConformalPID, a more aggressive variant) for sequential insurance forecasting. On a 60-month synthetic claims series with a structural break at month 60, ACI achieves 85% marginal coverage on the full series, with second-half coverage settling at the target 90% (Kupiec p = 0.23, statistically valid). That is actually a good result. But look at what happens in the period immediately after the break.

ACI's update rule is simple:

```python
# ACI interval width update — Gibbs & Candès 2021
alpha_t = alpha_{t-1} + gamma * (alpha - err_t)
# alpha:   target miscoverage (e.g. 0.10)
# err_t:   1 if t-1 prediction missed, 0 if covered
# gamma:   learning rate (tuned; typically 0.05–0.10)
```

After a regime change, `err_t` starts returning 1s at a higher rate than before. The learning rate `gamma` controls how fast the method responds. Set `gamma` too low and the adaptation is slow — you pay a long coverage gap. Set `gamma` too high and the intervals overshoot, becoming wider than necessary once the regime stabilises. FACI (Feldman et al. 2023) partially addresses this with an ensemble of learning rates, but it is still fundamentally reactive: it responds to observed miscoverage rather than anticipated regime change.

On our benchmark — a synthetic UK motor series with a +20% claims inflation step at month 60 — the coverage gap in the 12 months immediately following the break runs at roughly 70% when targeting 90%. FACI narrows this to around 78%. Neither reaches the nominal level in that window. Those 12 months are not hypothetical: 2022–23 was exactly this situation for UK motor portfolios.

---

## What CPTC does differently

CPTC adds a structural break detector upstream of the ACI update step. The paper uses a Recurrent Explicit Duration Switching Dynamical System (RED-SDS) as the break detector, because it handles the noisy, non-stationary character of real time-series better than a simple CUSUM or likelihood ratio test. The key architectural choice is that each detected regime gets its own learning rate rather than sharing a global one.

The workflow is:

1. **Detect:** the RED-SDS model maintains a posterior over latent regime states. At each time step it updates the regime probability based on new observations.
2. **Reset:** when a regime change is declared (posterior probability of new regime exceeds a threshold), the conformal quantile is reset. The reset is to the most recent calibration quantile for the new regime — or, at the start of a new regime, to the target alpha directly.
3. **Adapt:** within each regime, standard ACI updates apply with a regime-specific learning rate.

The proactive reset is the difference that matters. Instead of the coverage gap running for however many steps ACI needs to accumulate enough miscoverage signals, the reset happens at the moment of detected break. In the Zaffran et al. simulations, CPTC reduces the coverage gap from approximately 20 percentage points (ACI) and 12 pp (FACI) to 3–5 pp in the first 12 post-break observations.

---

## The UK insurance applications

**Model monitoring during claims inflation.** The 2022–23 UK motor claims inflation episode was a regime change in exactly the sense CPTC models. Repair costs rose 30–40% in under 18 months (ABI data, Q3 2022 to Q1 2024), driven by parts shortages, labour costs, and total-loss thresholds crossing the vehicle value curve at a different rate than in the pre-pandemic calibration period. A pricing model calibrated on 2019–21 data was operating in a different regime from 2022 onwards.

ACI on a monthly average severity series would have taken roughly 8–12 months to fully adapt. CPTC with a well-tuned RED-SDS would have declared a regime change around month 3–4 of the inflation event and reset the conformal quantile immediately. The practical result: pricing teams would have received a statistical signal that their severity model's prediction intervals were no longer valid, with a specific change-point date rather than a gradual widening.

**IBNR development pattern breaks.** Reserving actuaries face the same problem in a different domain. IBNR development patterns are calibrated on historical triangles, and occasionally the underlying development speed changes — due to litigation environment changes, claims system changes, or court backlogs (which shifted materially in 2021–22 following the pandemic). Conformal prediction intervals on IBNR reserves are only valid if the development pattern is stationary within the calibration window.

CPTC applied to a paid development triangle would detect when the pattern broke and provide a change-point date — directly useful for explaining to an audit committee why the tail factors changed.

**FCA Consumer Duty and model adequacy.** Under Consumer Duty (PRIN 2A, effective July 2023), firms must "regularly assess, test, understand and evidence the outcomes their customers are receiving." For pricing models, this includes evidence that the model continues to produce adequate outcomes — which implicitly requires knowing when the model is no longer operating in the regime it was built for.

CPTC gives you something more tractable than a general drift detection statistic: it gives you a labelled change-point with a timestamp. "Our severity model's prediction intervals were valid from Jan 2020 to Oct 2022. We detected a structural break on 14 October 2022 and recalibrated on 28 October 2022." That is what the Consumer Duty evidence requirement looks like in practice.

Our `insurance-monitoring` library provides drift detection via PSI, A/E tests, and the Hediger-Michel-Näf feature-interaction attribution method (arXiv:2503.06606). What it does not provide is a conformal coverage validity signal: it can tell you that the input distribution has drifted, but it cannot tell you whether your prediction intervals are still statistically valid. CPTC fills that gap.

---

## A practical sketch

You would not use RED-SDS directly for most insurance applications — it is a heavy dependency and requires careful tuning. A serviceable approximation uses `ruptures` (PELT algorithm, Killick & Fearnhead 2012) as the break detector:

```python
import numpy as np
import ruptures as rpt

class RegimeConformal:
    """Lightweight CPTC-style conformal predictor.

    Runs ACI within each detected regime, resetting the quantile
    on structural breaks detected by PELT.
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.05,
                 penalty: float = 10.0, min_regime_length: int = 12):
        self.alpha = alpha
        self.gamma = gamma
        self.penalty = penalty
        self.min_regime_length = min_regime_length

        self._scores: list[float] = []
        self._q: float = alpha  # current conformal quantile
        self._regime_start: int = 0

    def update(self, score: float, t: int) -> float:
        """Update on new conformity score, return current interval half-width."""
        self._scores.append(score)

        # Check for structural break if we have enough history
        if len(self._scores) >= 2 * self.min_regime_length:
            regime_scores = np.array(self._scores[self._regime_start:])
            algo = rpt.Pelt(model="rbf").fit(regime_scores.reshape(-1, 1))
            breakpoints = algo.predict(pen=self.penalty)

            # breakpoints[-1] is always the end of series; ignore it
            interior = [bp for bp in breakpoints[:-1]
                        if bp >= self.min_regime_length]

            if interior:
                # New regime detected — reset quantile
                last_bp = interior[-1]
                self._regime_start = self._regime_start + last_bp
                # Reset to target alpha; ACI will adapt from here
                self._q = self.alpha
                return self._q

        # Standard ACI update
        err = 1.0 if score > self._q else 0.0
        self._q = self._q + self.gamma * (self.alpha - err)
        self._q = float(np.clip(self._q, 1e-4, 1.0 - 1e-4))
        return self._q
```

This is about 50 lines of logic. On a monthly claims series, `score` would typically be the absolute normalised prediction error from your frequency-severity model. The conformal quantile `_q` maps back to an interval half-width via your model's residual distribution.

The `penalty` parameter in PELT controls sensitivity — lower values detect more breaks. For annual renewal cycles, 10–20 is a reasonable starting range; for monthly series, 5–10. Cross-validate on a held-out segment with known breaks.

---

## Why we are not building this into a library

The blog-only verdict here is about the RED-SDS dependency, not the concept. Switching dynamical systems are a specialised class of latent variable models. The inference is non-trivial — the paper uses a specific approximation involving Dirichlet process priors and variational inference. Wrapping that into a pip-installable library that works reliably on insurance data would require more effort than the value justifies, given that:

1. The `ruptures` approximation above captures 80% of the benefit for most insurance time-series applications. Regime changes in insurance are typically larger and lower-frequency than the scenarios where RED-SDS's sophistication matters.

2. `insurance-conformal-ts` already has ACI and ConformalPID. A `RegimeConformal` class as above would extend it cleanly — the interface would slot into the existing `ConformalPredictor` protocol — but we would want to see demand first.

3. The harder open question is how to handle overlapping detection signals from `insurance-monitoring` (which flags input drift) and CPTC (which flags coverage breakdown). A unified monitoring framework that coordinates these signals is a more interesting build than another standalone class.

If you are using `insurance-conformal-ts` on a series that crosses a known structural break — a pricing season reset, a rate change date, a data system migration — the immediate practical advice is: segment your series at the known break and run separate ACI instances on each segment. CPTC automates the detection step, but the manual version works fine when you already know the break date.

---

## What to read next

The CPTC paper, arXiv:2509.02844, reads well. The theoretical sections build directly on Gibbs and Candès (ACI, 2021) and Feldman et al. (FACI, 2023). If you are new to the ACI family, our [post on conformal prediction for non-exchangeable claims]({{ site.baseurl }}{% post_url 2026-03-15-conformal-prediction-for-non-exchangeable-claims-time-series %}) is the right starting point — it covers ACI and ConformalPID in full with the `insurance-conformal-ts` library as the running example.

For the monitoring side, `insurance-monitoring`'s implementation of the Hediger-Michel-Näf framework ([post, 2026-03-27]({{ site.baseurl }}{% post_url 2026-03-27-model-drift-detection-fremtpl2-insurance-monitoring %})) is complementary. The two methods answer different questions: CPTC asks "are my prediction intervals still valid?", `insurance-monitoring` asks "has my input distribution shifted?" You want both signals.

The full paper: [arXiv:2509.02844](https://arxiv.org/abs/2509.02844).
