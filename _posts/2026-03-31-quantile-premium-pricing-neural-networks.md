---
layout: post
title: "Setting premiums at the 85th percentile: quantile premium pricing with neural networks"
date: 2026-03-31
categories: [pricing, machine-learning]
tags: [quantile-regression, premium-pricing, QPP, QRNN, CatBoost, insurance-quantile, two-part-model, frequency-severity, FCA-Consumer-Duty, Solvency-UK, IFRS-17, Heras-2018, Zanzouri-2025, safety-loading, risk-appetite]
description: "The quantile premium principle maps a single number — your risk appetite parameter tau — to per-risk safety loadings. Zanzouri et al. (NAAJ 2025) shows QRNN outperforms tree-based quantile regression in the severity step. We explain the QPP formula, show the insurance-quantile implementation, and argue CatBoost should have been in their benchmark."
author: burning-cost
---

Most actuarial pricing models produce an expected loss estimate. The premium is then that estimate plus a loading: a flat percentage, a credibility-weighted margin, something derived from underwriting judgement. The loading has no formal connection to an explicit risk appetite level. If the CFO asks "what confidence level does our current loading correspond to?", the honest answer is usually "we don't know exactly."

The quantile premium principle (QPP) fixes this. You set a target aggregate confidence level $\tau$ — say 0.90 — and the framework derives, per risk, the exact safety loading implied by that confidence level and the risk's own claim frequency. The loading is not uniform: a high-frequency risk needs a smaller severity-side uplift to reach the 90th percentile than a low-frequency risk does. The framework does the translation automatically.

Heras, Moreno & Vilar-Zanon established the foundational formula in 2018 (*Scandinavian Actuarial Journal*, Vol. 2018, No. 9, pp. 753–769). Zanzouri, Kacem & Belkacem (NAAJ, June 2025) recently extended it to a systematic comparison of ML severity estimators. Their headline result — that QRNN outperforms quantile regression forest, gradient boosting, and XGBoost on an automobile dataset — is worth examining carefully, because the comparison has a notable omission.

---

## The QPP formula

Insurance losses are zero-inflated: most policies generate no claim. A standard quantile regression model fitted on all rows will produce zero estimates for any quantile below the zero mass, which for motor OD (where $\sim$80% of policies are claim-free) means anything below the 80th percentile is trivially zero. This is useless for pricing.

The QPP resolves this by decomposing the problem into frequency and severity steps, then mapping the aggregate target quantile to an adjusted severity-side quantile. The core formula is:

$$\tau_i = \frac{\tau - p_i}{1 - p_i}$$

where $\tau$ is your aggregate confidence level (0.90, 0.95, whatever you have chosen as policy), $p_i = P(N_i = 0 \mid x_i)$ is the no-claim probability for policy $i$, and $\tau_i$ is the severity quantile the model is asked to produce.

For a UK motor OD policy with $p_i = 0.80$ and a target of $\tau = 0.90$: $\tau_i = (0.90 - 0.80)/(1 - 0.80) = 0.50$. The severity model produces the *median* claim amount, not the 90th percentile of the severity distribution. The zero-inflation means the 90th percentile aggregate outcome corresponds to the 50th percentile severity outcome. This is why fitting a single quantile model to the full distribution fails — you need the decomposition.

The loaded premium is then a blend:

$$P_i = \gamma \cdot \tilde{Q}_{\tau_i}(x_i) + (1 - \gamma) \cdot E[S_i \mid x_i]$$

where $\tilde{Q}_{\tau_i}(x_i)$ is the conditional severity quantile at level $\tau_i$, $E[S_i \mid x_i]$ is the pure premium (expected loss), and $\gamma \in [0,1]$ is the loading weight. Setting $\gamma = 0$ gives the pure premium. Setting $\gamma = 1$ prices everything at the quantile. In practice $\gamma = 0.5$ is a reasonable default — half expected loss, half quantile.

The safety loading is the excess over expected loss:

$$\text{Loading}_i = \gamma \cdot \left( \tilde{Q}_{\tau_i}(x_i) - E[S_i \mid x_i] \right)$$

This loading is risk-specific, not uniform. A young driver in a high-frequency class has a lower $p_i$, so $\tau_i$ is pushed higher, and the severity quantile exceeds the mean by more. An older driver with fewer historical claims has a higher $p_i$, and the adjusted $\tau_i$ is lower — potentially even negative, in which case the QPP falls back to the pure premium (the formula requires $p_i < \tau$ to produce a valid $\tau_i$).

The QPP is a formal premium principle in the sense that it generalises the expected value premium principle: when $\tau \to p$ (you are pricing exactly at the zero-mass boundary), the loading goes to zero and $P_i \to E[S_i \mid x_i]$.

---

## What Zanzouri et al. (NAAJ 2025) actually show

The paper's contribution is modest but concrete: it is the first systematic ML benchmark for the severity step in the Heras 2018 framework. Specifically it compares four estimators for $\tilde{Q}_{\tau_i}(x_i)$:

- **QRF**: quantile regression forest (ranger-based)
- **QRGB**: quantile regression gradient boosting (LightGBM/XGBoost with quantile loss)
- **QRXGBoost**: XGBoost with quantile loss directly
- **QRNN**: quantile regression neural network (feedforward, pinball loss)

On the AutoClaims dataset (R package `insuranceData`, 6,773 closed private passenger auto claims from a US midwestern insurer, variables: STATE, CLASS, GENDER, AGE, target: PAID), QRNN wins on MSE, MAE, and pinball loss.

The underlying QPP framework — the $\tau_i$ formula, the two-part decomposition, the loaded premium blend — is unchanged from Heras 2018. The paper is substituting a better severity estimator into a known framework. That is useful, but it is not a new framework.

The notable omission from the benchmark is CatBoost. CatBoost's MultiQuantile loss function trains a single model across all quantile levels simultaneously (unlike XGBoost's per-quantile training), produces better-calibrated extreme quantiles, and typically outperforms LightGBM and XGBoost on structured tabular data with categorical features. The AutoClaims dataset has several categorical features (STATE, CLASS). We think CatBoost would have taken the top position in their benchmark. We have not run the full comparison to publication standard, but we consider the claim "QRNN is the best severity estimator for QPP" to be premature until CatBoost is included.

---

## Implementation in insurance-quantile

The full QPP pipeline is implemented in [insurance-quantile](https://github.com/burning-cost/insurance-quantile) as `TwoPartQuantilePremium`. The API takes a fitted frequency classifier (any sklearn-compatible estimator), a fitted `QuantileGBM` (CatBoost-backed multi-quantile model), and an optional mean severity model:

```python
from sklearn.linear_model import LogisticRegression
from insurance_quantile import QuantileGBM, TwoPartQuantilePremium

# Step 1: frequency model — fit on all policies
# y_freq = (claims > 0).astype(int)
freq_model = LogisticRegression(max_iter=500)
freq_model.fit(X_all.to_numpy(), y_freq)

# Step 2: severity model — fit on non-zero claims only
sev_model = QuantileGBM(
    quantiles=[0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
)
sev_model.fit(X_sev, y_sev_positive)

# Step 3: assemble the two-part model
tpqp = TwoPartQuantilePremium(freq_model, sev_model)

# Step 4: price at 90th percentile aggregate confidence, 50% loading weight
result = tpqp.predict_premium(X_val, tau=0.90, gamma=0.5)

print(result.premium.describe())
#  mean      312.40
#  std        87.23
#  min       189.00
#  25%       245.50
#  50%       298.00
#  75%       361.00
#  max       712.00

# Safety loading per policy
print(result.safety_loading.describe())

# Policies where tau_i was out of range (p_i >= tau)
print(f"Fallback count: {result.n_fallback}")
```

The `predict_premium` method applies the $\tau_i$ formula per policy, interpolates from the quantile grid using piecewise-linear interpolation across adjacent quantile levels, and returns a `TwoPartResult` with per-policy premiums, adjusted tau values, safety loadings, and a fallback counter for policies where $p_i \geq \tau$.

The fallback counter matters. If $p_i = 0.95$ and $\tau = 0.90$, there is no valid $\tau_i$ — the aggregate 90th percentile for this policy is already within the zero region of the loss distribution. The library falls back to the pure premium for these policies and counts them. A high fallback rate suggests $\tau$ is set too low relative to the typical claim frequency in your book, or that the frequency model is well-calibrated (most policies genuinely have low claim probability).

For the mean severity in the loading calculation, you can pass a dedicated mean model (recommended for production):

```python
from catboost import CatBoostRegressor

mean_sev = CatBoostRegressor(loss_function="Gamma", iterations=500, verbose=0)
mean_sev.fit(X_sev.to_pandas(), y_sev_positive)

tpqp = TwoPartQuantilePremium(freq_model, sev_model, mean_sev_model=mean_sev)
```

Without a dedicated mean model, the library integrates the quantile function via the trapezoid rule across the quantile grid. This is reasonably accurate with eight or more quantile levels but will underestimate the mean in the tails. A Gamma GLM or Tweedie GBM fitted on non-zero rows is a better choice.

---

## The tau parameter as a risk appetite instrument

The QPP framework's most useful property for UK regulatory purposes is that $\tau$ is an explicit, named risk appetite parameter. You can write in your Consumer Duty outcome monitoring report: "safety loadings are calibrated to ensure that for each risk segment, the loaded premium corresponds to the 90th percentile of the individual loss distribution." This is a precise, auditable statement. "We apply a 15% loading to expected loss" is not.

The FCA's Consumer Duty (PRIN 2A) requires that pricing provides fair value — price reasonable relative to benefits — and that loadings are causal (driven by risk characteristics, not exploitation of inertia or vulnerability). A per-risk quantile loading, where every pound of margin above expected loss is traceable to an explicit confidence level and the policy's own no-claim probability, directly satisfies the causal loading requirement in a way that flat percentage loadings cannot.

The Solvency UK SCR calculation (effective December 2024) is itself a quantile: 99.5th percentile one-year VaR of basic own funds. A `QuantileGBM` with extreme quantile extension can produce covariate-conditional 99.5th percentile estimates for individual risks — relevant for internal model capital allocation as well as premium setting. The same architecture supports both.

Under IFRS 17, the risk adjustment must be quantified as the compensation for bearing non-financial risk above the fulfilment cash flows. QPP provides a bottom-up, per-risk risk adjustment estimate grounded in a calibrated confidence level. That is more defensible under audit than a global percentage of best-estimate liability.

---

## A note on coherence

QPP is elicitable — the pinball loss is a consistent scoring function for quantiles — which is a necessary condition for meaningful model validation. But quantiles are not coherent in the sense of Artzner et al. (1999): they fail subadditivity. For a diversified portfolio, the sum of individual quantile premiums can exceed the portfolio quantile premium.

If you are pricing commercial lines where individual large losses dominate, or allocating capital to reinsurance layers, this matters. The expectile premium principle (EPP, Yang 2020, arXiv:2002.01798) is both elicitable and coherent — subadditive, so it plays well with portfolio diversification credit. `insurance-quantile` implements expectile mode alongside quantile mode for this reason.

For personal lines motor own damage — the context Zanzouri et al. test — subadditivity is not the binding constraint. Individual claims are bounded (vehicle replacement cost) and the portfolio is large. QPP is appropriate. For motor bodily injury or commercial property cat, think about whether coherence matters before defaulting to QPP.

---

## Where the paper's recommendation lands

Zanzouri et al. conclude that QRNN is the best severity model for the QPP pipeline on their dataset. We agree that QRNN is a strong choice: feedforward architectures with pinball loss train quickly, handle extrapolation reasonably, and benefit from the flexibility of neural architecture design (depth, dropout, batch normalisation for extreme quantile stability).

We disagree with the implicit conclusion that tree-based methods are inferior. CatBoost MultiQuantile was not in their comparison. On structured tabular data with categoricals — which describes nearly every UK insurance pricing dataset — CatBoost is a competitive baseline that typically beats vanilla gradient boosting implementations. The AutoClaims dataset has STATE and CLASS as categorical features. CatBoost handles ordered categoricals natively; XGBoost requires manual encoding.

Our default in `insurance-quantile` is CatBoost via `QuantileGBM`. Run it, check calibration at the quantile levels you care about, compare to QRNN if you have the infrastructure. If QRNN wins on your data, use QRNN — `TwoPartQuantilePremium` accepts any fitted quantile model with a `predict` interface. The QPP formula is the same either way.

The Heras 2018 paper remains the essential reference for the framework. Zanzouri 2025 adds a useful benchmark table. The library had the pipeline before the paper was published.
