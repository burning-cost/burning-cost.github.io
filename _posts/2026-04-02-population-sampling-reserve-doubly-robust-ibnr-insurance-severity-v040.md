---
layout: post
title: "Doubly Robust IBNR: The Middle Ground Between Chain-Ladder and Micro-Level Reserving"
date: 2026-04-02
categories: [techniques, reserving, actuarial]
tags: [reserving, ibnr, chain-ladder, micro-level, aipw, doubly-robust, weibull, inclusion-probability, solvency-ii, pra, motor, bodily-injury, missing-data, survey-statistics, insurance-severity, arXiv-2502.15598]
description: "PopulationSamplingReserve lands in insurance-severity v0.4.0. IBNR as a missing-data problem: the AIPW doubly-robust estimator hedges your bets between the chain-ladder's aggregate assumptions and the micro-level model's reporting assumptions. UK regulatory angle included."
math: true
author: burning-cost
---

Every reserving actuary eventually makes a version of the same choice: aggregate or individual?

Chain-ladder works on the triangle. It requires no claim-level data, needs no model for how claims are reported, and has been accepted by regulators and auditors for decades. Its weakness is that development factors assume stable patterns — the same mix of claim sizes and reporting behaviours in every accident period. When those patterns shift, CL produces biased estimates and you do not always know which direction.

Micro-level reserving takes individual claims as its unit of analysis. It can model the full development trajectory of each claim, handle heterogeneity in claim characteristics, and produce richer uncertainty estimates. Its weakness is that it requires an explicit model for how quickly claims get reported — the reporting delay distribution. If that model is misspecified, the IBNR count estimate is wrong, and the whole reserve is biased.

The choice feels forced. You can be wrong about claim severity, or you can be wrong about reporting speed. You have to pick one thing to trust.

Calcetero, Badescu and Lin at the University of Toronto (arXiv:2502.15598, February 2025) show that this is a false dilemma. Their key insight: IBNR is a missing-data problem. Claims not yet reported are "missing" from the sample. The statistical machinery for this situation — augmented inverse probability weighting (AIPW), a standard tool from survey statistics — produces a doubly-robust estimator. If either the reporting model or the severity model is correctly specified, the IBNR estimate is consistent. You only need one of the two to be right.

`insurance-severity` v0.4.0 ships `PopulationSamplingReserve`, implementing this framework.

---

## The missing-data framing

The setup is clean. At valuation time τ, we observe a sample of reported claims. Each claim incurred at time t_i has a reporting delay U_i — the time between incurral and notification. A claim is in our data only if U_i ≤ τ − t_i. The probability of this event is the **inclusion probability**:

$$\pi_i(\tau) = P(U_i \leq \tau - t_i \mid x_i)$$

This is the CDF of the reporting delay distribution evaluated at the maximum possible delay for that claim. For a claim incurred two years before valuation, there has been two years for it to arrive. For a claim incurred two months ago, there has been two months.

Claims with low π_i are likely IBNR. Claims with high π_i are almost certainly already reported. The total IBNR reserve is the sum of expected losses from all claims in the population that have not been reported yet.

The Horvitz-Thompson inverse probability weighted (IPW) estimator estimates the total:

$$\hat{L}_{\text{IBNR}}^{\text{IPW}} = \sum_{i \in \text{reported}} \frac{1 - \hat{\pi}_i}{\hat{\pi}_i} \cdot Y_i$$

This is unbiased if the inclusion probabilities are correctly specified. But it has high variance when π_i is small (weight blows up) and no way to incorporate claim-level covariates to improve precision.

The augmented IPW estimator (AIPW) adds a severity model Ŷ_i to reduce variance and correct for misspecification:

$$\hat{L}_{\text{IBNR}}^{\text{AIPW}} = \underbrace{\hat{n}_{\text{IBNR}} \cdot \bar{\hat{Y}}}_{\text{micro-level term}} + \underbrace{\sum_{i \in \text{reported}} \frac{1 - \hat{\pi}_i}{\hat{\pi}_i}(Y_i - \hat{Y}_i)}_{\text{augmentation term}}$$

The first term is what a pure micro-level model would give: the estimated count of unreported claims multiplied by the average predicted severity. The second term is the IPW-weighted residuals from the severity model over the reported claims — a bias correction that adjusts for the fact that reported claims are not a random sample of all claims.

**The double robustness property** (Proposition 2 of the paper): the AIPW estimate is consistent if either π̂_i is correctly specified OR Ŷ_i is correctly specified. Not both. Either one.

This is directly analogous to doubly-robust estimation in causal inference and survey statistics. It has been known in those fields since Scharfstein, Rotnitzky and Robins (1999). Calcetero et al.'s contribution is formalising its application to reserving and showing that chain-ladder, BF, Cape-Cod, and micro-level models are all special cases of this single framework.

---

## Chain-ladder is already here, hidden

The paper shows that standard chain-ladder is IPW with a specific choice of inclusion probabilities: for a claim in development period k, set π̂_i = 1/f_k, where f_k is the chain-ladder development-to-ultimate factor.

This is exact. The CL IBNR = Σ (f_k − 1) × Y_i = Σ (1/π̂_i − 1) × Y_i = Σ (1 − π̂_i)/π̂_i × Y_i.

Chain-ladder makes an implicit claim about reporting rates. It says: the probability that a claim incurred in accident period k has been reported by now is 1/f_k. When f_k = 2, the chain-ladder is implicitly assuming that only half the claims have been reported. When f_k = 1.05, it is assuming 95% are in.

Chain-ladder never makes this explicit. It works on aggregates, fits development factors, and the reporting rate interpretation is buried in the algebra. The AIPW framework makes it visible.

This matters practically because you can now ask: are my CL-implied reporting rates consistent with what the claim-level data tells me about reporting delays? If they are not, AIPW with a well-fitted Weibull inclusion model will outperform CL. If they are, both should give similar answers.

The `chain_ladder` method in `PopulationSamplingReserve` recovers CL exactly:

```python
from insurance_severity.reserving import PopulationSamplingReserve

# development_factors maps accident period label to f_k
dev_factors = {2022: 2.1, 2023: 1.6, 2024: 1.2}

psr = PopulationSamplingReserve(method="chain_ladder")
psr.fit(
    claims_df,
    development_factors=dev_factors,
    accident_time_col="accident_year",
)
psr.estimate_ibnr()   # identical to standard CL IBNR
```

This is verified in the test suite to machine precision (< 1e-9).

---

## WeibullInclusionModel: correcting for right truncation

The inclusion probability model is where most of the implementation work lives, and where the standard approach goes wrong.

Reporting delays are right-truncated. We only observe delays for claims that have been reported. Long delays are under-represented — claims with reporting delays longer than τ − t_i are not in the data at all. If you fit a Weibull (or exponential, or lognormal) distribution to observed reporting delays without accounting for this, you underestimate the scale parameter. The model thinks delays are shorter than they are on average, which means it overestimates inclusion probabilities, which means it underestimates IBNR.

The bias is not small on motor bodily injury data. UK BI claims can take 18 months or more to reach the insurer on a portfolio where 80% of claims come in within three months. Standard Weibull MLE on observed delays will see mostly the three-month claims and badly underestimate the 18-month tail.

The truncation-corrected log-likelihood divides each observation's contribution by the probability it was observable:

$$\ell(\theta) = \sum_{i \in \text{reported}} \left[ \log f(u_i \mid \theta, x_i) - \log F(\tau - t_i \mid \theta, x_i) \right]$$

The denominator F(τ − t_i) is the probability that a claim incurred at t_i would have been reported by τ. This downweights claims from recent accident periods (short observable windows) relative to older ones, correcting the bias.

`WeibullInclusionModel` fits this corrected likelihood via L-BFGS-B:

```python
from insurance_severity.reserving import WeibullInclusionModel
import numpy as np

# delay: observed reporting delay for each reported claim
# tau_minus_t: valuation_time - accident_time for each claim
delay = claims_df["report_date"] - claims_df["accident_date"]  # in months
tau_minus_t = valuation_date - claims_df["accident_date"]

model = WeibullInclusionModel(fit_covariates=False)
model.fit(delay.to_numpy(), truncation_times=tau_minus_t.to_numpy())

print(f"shape: {model.shape_:.3f}, intercept (log-scale): {model.intercept_:.3f}")
# shape: 1.42, intercept: 2.31  (mean delay ~exp(2.31) = 10 months)

# Inclusion probability for claims from 6, 12, 24 months ago
pi = model.predict_inclusion_prob(np.array([6.0, 12.0, 24.0]))
# [0.52, 0.78, 0.96]
```

With `fit_covariates=True` and a covariate matrix, the model fits a log-linear effect on the scale parameter. This allows reporting speed to vary by claim characteristic — useful if bodily injury claims from one distribution channel have systematically longer delays than another.

---

## Fitting AIPW end-to-end

The full workflow requires a DataFrame of reported claims with accident time, report time, and a severity column. `PopulationSamplingReserve` auto-fits a `WeibullInclusionModel` if no inclusion model is supplied, and uses lognormal moment estimates as a fallback severity model:

```python
from insurance_severity.reserving import PopulationSamplingReserve

# claims_df has columns: accident_time, report_time, severity
# Plus any feature columns you want to pass to the models

psr = PopulationSamplingReserve(method="aipw")
psr.fit(
    claims_df,
    valuation_time=2024.0,       # τ as a numeric (e.g., decimal year)
    feature_cols=["claim_age_months", "channel"],
)

print(f"IBNR estimate:    £{psr.estimate_ibnr():,.0f}")
print(f"Ultimate estimate: £{psr.estimate_ultimate():,.0f}")
```

The diagnostics tell you what happened under the hood:

```python
d = psr.diagnostics()
# {
#   'ibnr_estimate': 4_823_000.0,
#   'ultimate_estimate': 31_210_000.0,
#   'n_reported': 2847,
#   'n_ibnr_estimated': 412.3,       # HT estimate of IBNR count
#   'augmentation_term': 183_000.0,  # positive = severity model underestimates IBNR claims
#   'weighted_balance_ratio': 1.08,  # close to 1.0 = severity model is well-calibrated
#   'method': 'aipw',
#   'total_reported_losses': 26_387_000.0,
#   'valuation_time': 2024.0
# }
```

The augmentation term is the key diagnostic. If it is near zero, the severity model is approximately satisfying the weighted balance property — the AIPW and pure micro estimates will be close. If it is large relative to the IBNR estimate, the severity model is materially biased for IBNR claims and the augmentation is doing real work.

The weighted balance ratio b = Σ[(1−π̂)/π̂ × Y] / Σ[(1−π̂)/π̂ × Ŷ] should be close to 1.0 when the severity model is well-calibrated. Values materially above 1.0 indicate the severity model is underpredicting for recent (low-π) claims; below 1.0 indicates overprediction.

---

## Comparing all four estimators

Running all four variants on the same data is straightforward:

```python
results = {}
for m in ["aipw", "ipw", "micro", "chain_ladder"]:
    kwargs = {}
    if m == "chain_ladder":
        kwargs["development_factors"] = {2022: 2.1, 2023: 1.6, 2024: 1.2}
    if m == "micro":
        kwargs["n_ibnr"] = 410   # externally estimated or from another run

    obj = PopulationSamplingReserve(method=m)
    obj.fit(claims_df, valuation_time=2024.0, **kwargs)
    results[m] = obj.estimate_ibnr()

# method         IBNR estimate
# aipw           4,823,000
# ipw            4,691,000
# micro          5,106,000
# chain_ladder   4,540,000
```

The spread between estimates tells you something useful. Wide disagreement between AIPW and chain-ladder suggests the CL-implied reporting rates are inconsistent with the claim-level delay data. Wide disagreement between AIPW and micro suggests the severity model is not well-calibrated for the unreported population. When all four are close, you have convergent evidence that your assumptions are mutually consistent.

---

## The UK regulatory angle

Under Solvency II (and Solvency UK as now implemented by the PRA), the best estimate liability must be the mean of the distribution of future cash flows, with explicit assumptions about reporting delays for IBNR claims. The Solvency II best estimate requirements (and PRA model governance expectations under Solvency II Articles 120–126) require that the assumptions underlying any reserving model be documented, testable, and subject to sensitivity analysis. SS1/23 establishes equivalent principles for banks; it does not directly apply to insurers, but its documentation standards are widely adopted as best practice.

Chain-ladder satisfies this in practice because it is familiar and its assumptions are implicitly accepted by convention. But "familiar" is not the same as "defensible". The AIPW framework offers two regulatory advantages:

**Formal bias-reduction guarantees.** The doubly-robust property is a mathematical result, not a rule of thumb. If you can demonstrate that either your inclusion model or your severity model is well-specified (via diagnostics or validation on held-out data), you have a formal guarantee that the IBNR estimate is asymptotically unbiased. This is more defensible to a PRA validator than "we used chain-ladder with selected factors."

**Explicit reporting delay assumptions.** The `WeibullInclusionModel` requires you to state and fit a reporting delay distribution, with the truncation correction applied explicitly. This is a testable, falsifiable claim about your portfolio's reporting behaviour. You can validate it against historical experience, compare fitted parameters across accident years, and document the results.

UK motor bodily injury is the most compelling use case. BI claims are genuinely long-tailed on reporting: the pre-litigation period, treatment completion, and legal process can each add months to the delay. The right-truncation correction is material for BI claims from recent accident years, which are exactly the ones driving uncertainty in the current-year reserve. Using uncorrected Weibull MLE on BI delay data will systematically underestimate how many claims are still to come.

The augmentation term also has a BI-specific interpretation: large claims — catastrophic injuries, long-term care cases — tend to be notified later on average because they require more investigation before being registered. This is sampling bias in the classic sense. Reported BI claims are a biased sample of all BI claims, skewed toward less severe notifications. The augmentation term in AIPW corrects for this by re-weighting reported claims according to their probability of being representative of the unreported population.

---

## What it does not do

A few honest limitations.

The framework does not handle RBNS (reported but not settled) claims. `PopulationSamplingReserve` addresses only the IBNR component — unreported claims — not the development of open reported claims. For a complete reserving workflow you would combine IBNR estimation here with `ProjectionToUltimate` (also in `insurance-severity`) for open claims.

The auto-fitted `WeibullInclusionModel` assumes a parametric form for reporting delays. If your reporting delay distribution is genuinely non-parametric (multimodal, with distinct fast- and slow-reporting populations), the Weibull will underfit. You can supply any callable as `inclusion_model` — including a kernel density or a more flexible parametric family — and the AIPW machinery works unchanged.

The default severity model (lognormal moment estimates) is a fallback that will be wrong in most production reserving contexts. Supplying a well-fitted severity model — a GLM, or the `LognormalBurrComposite` from the same library — will sharpen the IBNR estimate and make the augmentation term meaningful rather than just a correction for an underfitted default.

---

## Installation

```
uv add insurance-severity>=0.4.0
```

The reserving module is pure numpy/scipy — no PyTorch dependency. It is in `insurance_severity.reserving` and also importable from the top level:

```python
from insurance_severity import PopulationSamplingReserve, WeibullInclusionModel
```

**Paper:** [arXiv:2502.15598](https://arxiv.org/abs/2502.15598) — Calcetero-Vanegas, Badescu & Lin (2025) | **Library:** [insurance-severity on PyPI](https://pypi.org/project/insurance-severity/) | **GitHub:** [insurance-severity](https://github.com/burning-cost/insurance-severity)
