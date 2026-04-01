---
layout: post
title: "QPP ratemaking: when quantile loading genuinely helps, and the exposure offset gap"
date: 2026-04-01
categories: [pricing, techniques]
tags: [quantile-regression, QPP, quantile-premium-principle, TwoPartQuantilePremium, insurance-quantile, frequency-severity, two-part-model, CompoundPoissonQPP, exposure-offset, Poisson-GLM, heavy-tails, coherence, FCA-Consumer-Duty, IFRS-17, motor-OD, motor-BI, flood, Heras-2018, Zanzouri-2025, python]
description: "A follow-up to our QPP introduction: the honest case for quantile-based loading (it works for heavy-tailed lines and low-frequency risks, it does not work below the zero mass), the correct UK tau values by line of business, and the one genuine implementation gap — CompoundPoissonQPP for policies with varying exposure."
math: true
author: burning-cost
---

[We covered the QPP formula and the Zanzouri 2025 benchmark in a previous post.](https://burning-cost.github.io/pricing/machine-learning/2026/03/31/quantile-premium-pricing-neural-networks/) This post is the harder question: when does pricing at the $\tau$-th percentile actually change what you charge, relative to the expected value premium principle — and when is it window-dressing?

The short answer: QPP adds genuine value when the severity distribution is skewed enough that the adjusted severity quantile $\tilde{Q}_{\tau_i}(x_i)$ sits materially above the conditional mean. For a near-symmetric severity distribution, the quantile and the mean are close, the loading is modest, and QPP converges toward the expected value premium principle with a small upward nudge. For a heavy-tailed line — flood property, motor bodily injury, professional indemnity — the quantile can be two or three times the mean, and the per-risk loading is substantial. That difference is where the framework earns its complexity.

There is also an exposure offset gap in the current implementation that affects any book with policies of varying duration. We explain what it is and why it matters before describing the straightforward fix.

---

## What the Expected Value Premium Principle actually does

The expected value premium principle sets:

$$P_i = (1 + \theta) \cdot E[S_i \mid x_i]$$

where $\theta$ is a flat relative loading, the same for every policy. The loading is uniform by design: every pound of expected loss attracts the same proportional margin, regardless of how dispersed or concentrated that risk is.

This is actuarially defensible for homogeneous lines with stable loss ratios. It becomes harder to defend for two kinds of policy: very low-frequency risks (where the expected loss is a tiny number and the realistic loss if anything happens is large), and risks where the severity distribution has a fat right tail (where the expected loss dramatically understates the upside exposure).

QPP addresses both. For a low-frequency risk with $p_i = 0.95$ (5% claim probability) and $\tau = 0.90$: $\tau_i = (0.90 - 0.95)/(1 - 0.95)$ is negative — QPP tells you there is no valid quantile-based loading at this confidence level, because the 90th percentile aggregate outcome is zero for this risk. The framework correctly falls back to the pure premium. This is not a limitation; it is mathematically honest. The expected value principle would apply a flat uplift regardless.

For a mid-frequency risk with $p_i = 0.70$ and $\tau = 0.90$: $\tau_i = (0.90 - 0.70)/0.30 = 0.67$. The severity model is asked for the 67th percentile of the conditional claim amount. For a log-normal severity distribution with $\mu = 7$ (£1,097 median) and $\sigma = 1.2$, the 67th percentile is approximately £1,860 and the conditional mean is approximately £2,250 (the mean exceeds the median for log-normal). Here $\tilde{Q}_{0.67}$ is *below* the conditional mean, so the QPP loading at $\gamma = 0.5$ is actually negative. The premium is lower than the expected value premium principle would give. This is correct behaviour — QPP is a VaR-based pricing approach that respects the actual distributional shape, and a moderate quantile on a right-skewed distribution lands below the mean.

For a high-dispersion risk with $\sigma = 2.5$ (heavier tail), the same 67th percentile is around £3,300 against a conditional mean of approximately £25,000 — the extreme skew of the distribution pushes the mean far above any moderate quantile. QPP would give a substantially lower loaded premium than the EVP for this risk at $\tau = 0.90$ with $p_i = 0.70$. This is the regime where QPP and EPP most sharply diverge: the flat EPP uplift would be calibrated to the mean (£25,000), while QPP at this $\tau$ targets only the 67th percentile of severity (£3,300). If you want a higher loaded premium via QPP, you need a higher aggregate $\tau$ — closer to 0.98 — so that $\tau_i$ clears the severity mean.

To get a positive QPP loading from a skewed severity distribution, you need $\tau_i$ to land above the percentile of the severity distribution at which the quantile equals the conditional mean. For lognormal severity, that percentile is $\Phi(\sigma/2)$: roughly 0.73 for $\sigma = 1.2$ and 0.89 for $\sigma = 2.5$. The required aggregate $\tau$ then follows from inverting the QPP formula: $\tau > p_i + \Phi(\sigma/2) \cdot (1 - p_i)$. For $p_i = 0.70$ and $\sigma = 1.2$, that requires $\tau > 0.92$. For heavier tails ($\sigma = 2.5$), $\tau > 0.97$. The practical implication: for standard UK motor OD severity (log-normal, moderate dispersion), you need to price at $\tau = 0.92$+ to get a positive QPP loading. Below that, QPP produces a lower loaded premium than the flat expected value uplift.

---

## Where QPP genuinely earns its keep

The framework adds real value in three distinct scenarios.

**Heavy-tailed severity.** When the severity coefficient of variation is high — motor BI, flood-exposed property, professional indemnity — the conditional mean is dominated by rare large losses. The quantile at a high $\tau_i$ can be several multiples of the mean. QPP translates this into a risk-specific loading: a policy where the severity 90th percentile is £50,000 against a mean of £8,000 gets a loading proportional to £42,000 of excess; a policy where the 90th percentile is £15,000 against a mean of £12,000 gets a loading proportional to £3,000 of excess. The expected value principle would give the same relative uplift to both.

**Low-frequency lines where any claim is a large loss.** For surety bonds, professional indemnity, or financial guarantee — where claim probability per policy is below 5% but any individual claim exceeds £100,000 — QPP correctly identifies that a high aggregate confidence level maps to a very high conditional severity quantile. The framework enforces that pricing reflects this; flat percentage loadings applied to small expected values can produce premiums that are inadequate against tail exposure.

**UK regulatory documentation.** The QPP framework has a specific transparency advantage: $\tau$ is a named risk appetite parameter that can be written into product governance documentation, Consumer Duty outcome monitoring reports, and IFRS 17 risk adjustment notes. "Safety loadings are calibrated to the 90th percentile of each risk's individual loss distribution" is a precise, auditable statement. The loading is not arbitrary; it traces directly to the frequency model output ($p_i$) and the severity quantile model output ($\tilde{Q}_{\tau_i}$). Under FCA Consumer Duty, per-risk causal loadings are substantially easier to defend than uniform uplift factors.

---

## Where QPP does not help

**Below the zero mass.** Any risk where $p_i \geq \tau$ has a QPP loading of zero. If 95% of your policyholders are claim-free and you are pricing at $\tau = 0.90$, QPP applies no loading to any policy. This is mathematically correct — the 90th percentile aggregate loss for any of these policies is zero — but it means QPP is entirely silent below the zero mass. You still need a separate decision about how to charge the 5% of risks that do claim. QPP does not price that exposure; it only handles the severity-conditional loading for risks above the threshold.

**Symmetric or thin-tailed severity.** For lines where severity is close to normal or where the coefficient of variation is low, the quantile and the mean are similar across the range of plausible $\tau_i$ values. QPP produces loadings that are barely different from a uniform expected value uplift. The added complexity of fitting and calibrating a quantile model is not repaid.

**Portfolio-level coherence.** QPP is not coherent in the sense of Artzner, Delbaen, Eber & Heath (1999). Quantile-based risk measures fail subadditivity: the sum of $\text{VaR}_{0.95}$ across individual risks can exceed $\text{VaR}_{0.95}$ of the aggregate portfolio. For reinsurance pricing where individual large losses dominate, or for capital allocation across a diversified book, this matters. The expectile premium principle is both elicitable and coherent; `QuantileGBM` in expectile mode is the right tool for those use cases. For personal lines motor own damage — where the portfolio is large, individual losses are bounded by vehicle value, and diversification effects are strong — the coherence issue is not binding.

---

## UK lines: which tau to use

The following is our empirical guidance based on the mathematical structure of QPP, published calibration studies in the actuarial literature, and characteristics of UK insurance lines. None of these are absolute; calibrate against your own portfolio.

**Motor own damage** ($p_i \approx 0.80$, lognormal severity, bounded by vehicle value): $\tau = 0.93$–$0.97$. Note that at $\tau = 0.90$, $\tau_i = 0.50$ and for typical UK OD severity ($\sigma \approx 1.2$) the severity median sits *below* the conditional mean — QPP would produce a slight negative loading. To get a positive loading at $p_i = 0.80$, you need $\tau \geq 0.95$ (so that $\tau_i \geq \Phi(0.6) \approx 0.73$, which is the percentile at which the lognormal quantile equals the mean). Loadings are modest given bounded vehicle-value severity; the main benefit is the auditable confidence level, not the loading magnitude.

**Motor third party property damage** ($p_i \approx 0.85$, similar severity to OD but occasional large bodily damage claims): $\tau = 0.92$–$0.97$. The slight additional tail exposure from third-party property justifies a higher aggregate confidence level.

**Motor bodily injury** (heavy tail, possible multi-million-pound claims): $\tau = 0.95$–$0.99$, but note the coherence concern above. For BI pricing, we prefer expectile mode in `QuantileGBM` — the expectile premium principle gives a risk-specific loading that is coherent and handles the heavy tail better. QPP is usable but not our first choice for BI.

**Domestic property** (frequency varies by peril; flood-exposed risks have $p_i$ as low as $0.50$ in high-flood zones): $\tau = 0.95$–$0.99$ for flood-exposed risks. For non-flood property OD, $\tau = 0.90$–$0.95$. QPP is appropriate; the adjusted $\tau_i$ for flood-exposed risks will land in the high-severity tail where the quantile model has the most information.

**Commercial property / SME** (higher severity, lower frequency per policy): $\tau = 0.95$+. For thin segments within commercial lines, `WassersteinRobustQR` (v0.4.0) is the right severity quantile estimator — but applied directly to the segment, not as a drop-in replacement for `QuantileGBM` inside `TwoPartQuantilePremium`.

---

## The exposure offset gap

The current `TwoPartQuantilePremium` implementation assumes a binary frequency model: the no-claim probability $p_i$ comes from a logistic classifier trained on an indicator variable $\mathbf{1}(N_i > 0)$. This works when every policy in the training data has the same exposure period — typically one full policy year.

Most UK pricing datasets do not have uniform exposure. Mid-term adjustments, short-term policies, new policies incepting partway through the year, and multi-year policies all produce varying exposure periods. The correct frequency model for this data is a Poisson GLM with an offset:

$$\log(\lambda_i) = x_i^\top \beta, \qquad N_i \sim \text{Poisson}(\lambda_i \cdot e_i)$$

where $e_i$ is the exposure (fraction of a policy year). The no-claim probability is then:

$$p_i = P(N_i = 0) = e^{-\lambda_i \cdot e_i}$$

This is the entry point into the QPP formula. The formula $\tau_i = (\tau - p_i)/(1 - p_i)$ is unchanged — it only requires $p_i$, and the Poisson GLM produces exactly the right $p_i$ once exposure is accounted for. The mathematical structure of QPP is agnostic to how you model frequency; the key step is getting the right no-claim probability for each policy.

The current `TwoPartQuantilePremium` does not have a native path for this. The `freq_model` parameter accepts any sklearn-compatible classifier with a `predict_proba` interface, but `sklearn.linear_model.PoissonRegressor` does not implement `predict_proba` — it predicts the Poisson rate $\lambda_i$, from which you compute $p_i = e^{-\lambda_i \cdot e_i}$ separately.

The gap is not deep. A `CompoundPoissonQPP` class would wrap this conversion:

```python
class CompoundPoissonQPP:
    """
    Two-part QPP with Poisson frequency model and exposure offset.

    For Poisson(lambda_i * exposure_i) claim frequency:
        p_i = exp(-lambda_i * exposure_i)
        tau_i = (tau - p_i) / (1 - p_i)   # unchanged QPP formula

    Parameters
    ----------
    lambda_model:
        A fitted model with .predict(X) -> lambda_i (expected claim rate per
        unit exposure). PoissonRegressor, CatBoostRegressor with Poisson loss,
        or any positive-output regressor.
    sev_model:
        A fitted QuantileGBM trained on non-zero claims.
    mean_sev_model:
        Optional. A fitted mean severity model. If None, trapezoidal
        approximation from the QuantileGBM quantile grid is used.
    """
    def predict_premium(
        self,
        X: pl.DataFrame,
        exposure: pl.Series,
        tau: float = 0.95,
        gamma: float = 0.5,
    ) -> TwoPartResult:
        lambda_i = self._lambda_model.predict(X)
        p_i = np.exp(-lambda_i * exposure.to_numpy())
        # ... remainder identical to TwoPartQuantilePremium from here
```

This is approximately 100–150 additional lines in `_two_part.py`, with no new dependencies. It is the correct implementation for any UK motor or property book where policies are not all full-year.

We have not built this yet. It is on our list for insurance-quantile v0.5.0.

---

## QPP vs the Expected Value Premium Principle: a summary table

| Scenario | EPP (flat loading) | QPP |
|---|---|---|
| Homogeneous frequency, symmetric severity | Adequate | Minimal improvement |
| Heavy-tailed severity (BI, flood, PI) | Loading understates tail | Risk-specific loading from severity quantile |
| Low-frequency book ($p_i > \tau$) | Applies loading regardless | Correctly returns pure premium |
| Very low frequency, very large claims (surety, FG) | Flat uplift on small expected loss | High $\tau_i$, large per-risk loading |
| Varying policy duration | Logistic with no offset | Needs CompoundPoissonQPP (not yet built) |
| Portfolio coherence required | Subadditive (flat) | Not coherent — use expectile instead |
| UK regulatory audit trail | "We apply a 15% loading" | "$\tau$ = 0.90, per-risk via Heras 2018 formula" |

---

## What the Zanzouri 2025 paper tells us about all this

[Zanzouri, Kacem & Belkacem (NAAJ, June 2025)](https://doi.org/10.1080/10920277.2025.2503744) benchmarks four ML severity estimators on the AutoClaims dataset — 6,773 closed US auto claims from the 1990s, four features (STATE, CLASS, GENDER, AGE). QRNN wins on MSE, MAE, and pinball loss.

The paper is useful as a benchmark table but says nothing about the three substantive questions above: when does the loading matter, what $\tau$ to use for which line, or how to handle exposure offsets. The dataset has no exposure variable, uses GENDER (unusable for UK pricing under the Gender Directive), and its severity distribution is approximately lognormal with moderate tail index — not the heavy-tailed case where QPP most clearly outperforms EPP.

CatBoost was not included in their benchmark. On structured tabular data with categorical features — which describes every UK motor book — CatBoost MultiQuantile typically performs at or above LightGBM and XGBoost with quantile loss, and above QRF on datasets of this size. The "QRNN is the best severity model for QPP" claim should not be taken as settled.

Our default in `insurance-quantile` is CatBoost via `QuantileGBM`. The QPP formula is identical regardless of which severity quantile estimator you use.

---

## The implementation

`TwoPartQuantilePremium` is in [`insurance-quantile`](https://github.com/burning-cost/insurance-quantile), v0.3.1+. Full usage in the [earlier post](https://burning-cost.github.io/pricing/machine-learning/2026/03/31/quantile-premium-pricing-neural-networks/).

```bash
pip install "insurance-quantile>=0.3.1"
```

```python
from insurance_quantile import QuantileGBM, TwoPartQuantilePremium
```

For thin segments — commercial, high-net-worth, agricultural — where per-rating-cell claim counts fall below a few hundred, `WassersteinRobustQR` (v0.4.0) handles the severity quantile directly. It does not drop into `TwoPartQuantilePremium` as a `sev_model` replacement: `TwoPartQuantilePremium` expects a `QuantileGBM` with a full multi-level quantile grid, and interpolates across that grid per policy. `WassersteinRobustQR` is a linear model that estimates a single quantile level with a finite-sample robustness guarantee. The two tools address different problems. For a thin commercial segment, use `WassersteinRobustQR` to estimate the severity quantile for your chosen $\tau_i$, then apply the QPP formula manually. See the [WassersteinRobustQR post](https://burning-cost.github.io/techniques/pricing/2026/04/01/wasserstein-robust-quantile-regression-thin-insurance-portfolios/) for the detail.

---

## References

Heras, A., Moreno, I., & Vilar-Zanon, J.L. (2018). An application of two-stage quantile regression to insurance ratemaking. *Scandinavian Actuarial Journal*, 2018(9), 753–769. DOI: 10.1080/03461238.2018.1452786.

Zanzouri, S., Kacem, M., & Belkacem, L. (2025). Insurance Ratemaking Using a Combined Quantile Regression Machine Learning Approach. *North American Actuarial Journal*. DOI: 10.1080/10920277.2025.2503744.

Laporta, A.G., Levantesi, S., & Petrella, L. (2024). Neural networks for quantile claim amount estimation. *Annals of Actuarial Science*, 18(1), 30–50. DOI: 10.1017/S1748499523000106.

Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.

Yang, L. (2020). Using expectile regression for classification ratemaking. arXiv:2002.01798.

---

## Related posts

- [Setting premiums at the 85th percentile: quantile premium pricing with neural networks](https://burning-cost.github.io/pricing/machine-learning/2026/03/31/quantile-premium-pricing-neural-networks/) — QPP formula, Zanzouri benchmark, full API
- [The 99th Percentile From 200 Claims: Wasserstein Robust Quantile Regression for Thin Portfolios](https://burning-cost.github.io/techniques/pricing/2026/04/01/wasserstein-robust-quantile-regression-thin-insurance-portfolios/) — `WassersteinRobustQR` as the severity estimator in thin segments
- [insurance-quantile: Tail Risk Quantile and Expectile Regression for Insurance](https://burning-cost.github.io/techniques/pricing/2026/03/07/insurance-quantile/) — library overview
