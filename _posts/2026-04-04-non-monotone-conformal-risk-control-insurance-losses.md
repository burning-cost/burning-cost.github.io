---
layout: post
title: "Conformal Risk Control Assumes Your Loss Decreases With Interval Width. Most Insurance Losses Don't."
date: 2026-04-04
categories: [conformal-prediction, insurance-pricing]
tags: [conformal-risk-control, non-monotone-loss, interval-scoring, winkler-score, solvency-ii, finite-sample-guarantee, insurance-conformal, aldirawi-2026, arXiv-2604.01502, model-validation, uncertainty-quantification, actuarial]
description: "Conformal risk control (Angelopoulos et al. ICLR 2024) requires monotone loss functions for its finite-sample guarantees. The Winkler score, two-sided regulatory tests, and capital requirement functions are all non-monotone. Aldirawi, Li and Guo (arXiv:2604.01502, April 2026) generalise CRC to this setting with an O(sqrt(log(m)/n)) bound. We explain what breaks without monotonicity and why this matters for insurance model validation."
math: true
author: burning-cost
---

Conformal risk control is built on a quiet assumption that most users never examine: the loss function must decrease monotonically as the prediction set grows. Make the interval wider, the loss goes down (or stays flat). That assumption is what makes the guarantee go through.

A lot of insurance losses violate it.

The Winkler interval scoring rule penalises both misses and unnecessary width — a wider interval is not always better. Two-sided solvency tests require predictions to be neither too low nor too high — a prediction that moves in the wrong direction can increase loss even as the set expands. Capital requirement functions can be non-monotone in their inputs for structural reasons specific to how SCR is computed under Solvency II. A concrete example: under the standard formula, SCR is computed by aggregating risk modules using a correlation matrix. If one risk module (say, market risk) has a negative correlation with a dominant module (say, underwriting risk), then increasing the market risk module's standalone capital charge can reduce the aggregate SCR — because the diversification benefit from the negative correlation grows faster than the standalone charge. The mapping from individual module inputs to aggregate SCR is therefore non-monotone. These are not edge cases.

Aldirawi, Li and Guo (arXiv:2604.01502, April 2026) drop the monotonicity requirement and derive finite-sample guarantees for conformal risk control over a finite grid of tuning parameters. The excess risk above target is bounded at O(√(log(m)/n)) where m is the grid size and n is the calibration sample size. They show this rate is minimax optimal — no method can do better in general.

---

## Why monotonicity matters in standard CRC

The Angelopoulos et al. (ICLR 2024) conformal risk control procedure works roughly as follows. You have a family of prediction sets $C_\lambda(X)$ indexed by a tuning parameter $\lambda$. As $\lambda$ increases, the sets grow. The loss $L(C_\lambda(X), Y)$ is assumed to decrease in $\lambda$: a bigger set is better. You calibrate by finding the largest $\lambda$ on a held-out set such that the empirical risk stays below target $\alpha$.

The monotonicity assumption does real work here. It ensures the calibration objective is well-behaved: the empirical risk function in $\lambda$ has a single crossing point, so the calibration procedure finds a sensible threshold. It also controls the relationship between the chosen $\lambda$ and the expected risk on new data — the key step in the finite-sample guarantee.

Take away monotonicity and the calibration landscape becomes irregular. The empirical risk in $\lambda$ can have multiple local minima. The naive procedure might select a $\lambda$ that looks good on calibration data but happens to sit in a trough that does not generalise. The guarantee breaks.

---

## The insurance losses that are non-monotone

Three loss functions that appear regularly in insurance model evaluation are non-monotone in interval width.

**The Winkler interval score.** For a prediction interval $[\hat{l}, \hat{u}]$ with target coverage $1 - \alpha$, the Winkler score for a single observation $y$ is:

$$S(\hat{l}, \hat{u}, y) = (\hat{u} - \hat{l}) + \frac{2}{\alpha} \cdot ((\hat{l} - y)^+ + (y - \hat{u})^+)$$

The first term penalises interval width directly. The second term penalises misses. This is the right trade-off for reporting prediction intervals to management — you should not be rewarded for saying "the claim will be between £0 and £∞". But it creates non-monotonicity: widening the interval reduces the miss penalty but increases the width penalty. The loss is minimised at some interior point, not by taking $\lambda \to \infty$.

The Winkler score is used in the Makridakis M-competitions and is increasingly used in actuarial model comparison. If you are using CRC to calibrate a prediction set to minimise expected Winkler score — a natural goal — standard CRC does not apply.

**Two-sided regulatory tests (reserve and capital models).** Solvency II requires that reserve estimates be adequate — not too low. That is the standard one-sided constraint that monotone CRC handles straightforwardly: make the interval upper bound high enough.

But internal model validation under Solvency II Article 121 involves two-sided criteria in practice for reserve and capital models, not for pricing. A reserve or capital model that consistently over-estimates by a large margin is no more defensible than one that under-estimates — a capital model that doubles the SCR is not acceptable any more than one that halves it. (Pricing models have a different loss structure: the asymmetry between underpricing and overpricing is commercial and competitive, not a two-sided regulatory test of the same form.) Formally, for a reserve or capital model you might define a loss that penalises both under-estimation (regulatory risk) and over-estimation (capital inefficiency):

$$L(\lambda) = \max\left(\frac{\alpha_{\text{target}} - \hat{r}(\lambda)}{\alpha_{\text{target}}}, 0\right) + \beta \cdot \max\left(\frac{\hat{r}(\lambda) - \alpha_{\text{target}}}{\alpha_{\text{target}}}, 0\right)$$

where $\hat{r}(\lambda)$ is the estimated risk at tuning parameter $\lambda$ and $\beta$ is the relative weight on over-estimation. This is non-monotone: as you increase $\lambda$, the first term decreases but the second can increase.

**Premium sufficiency with margin constraints.** The `PremiumSufficiencyController` in `insurance-conformal` controls expected shortfall — the one-sided loss $E[\max(y - \lambda p, 0)/p] \leq \alpha$. This is monotone in $\lambda$: a higher loading factor reduces shortfall. But regulators and capital optimisers both care about the two-sided version: not just that the loading factor is high enough to avoid shortfall, but that it is not so high that it prices the firm out of market. A cost-of-capital objective that trades off underprice risk against overcharge risk is, again, non-monotone in $\lambda$.

---

## What the paper does

Aldirawi, Li and Guo work on a finite grid of tuning parameters $\{\lambda_1, \ldots, \lambda_m\}$, which is the natural setting for practical implementations — you discretise the search space, evaluate the empirical risk at each grid point, and select the best $\lambda$.

Their main result: for bounded losses over a grid of size $m$ with calibration sample $n$, the excess risk of the selected $\lambda$ above the target $\alpha$ is bounded with high probability at:

$$\text{ExcessRisk} \leq C \sqrt{\frac{\log m}{n}}$$

This is minimax optimal — they prove a matching lower bound, so no method can achieve a uniformly better rate without additional assumptions. The proof does not require monotonicity.

The construction is simpler than you might expect. The key is the choice of calibration statistic: rather than relying on monotonicity to ensure the calibration objective is single-crossing, they work directly with uniform concentration inequalities over the finite grid. The Bonferroni-type argument — the probability that any of the $m$ grid points has an empirical risk deviating from its expectation by more than $\epsilon$ is at most $m$ times the per-grid-point probability — combined with a union bound over the grid gives the $\sqrt{\log m}$ factor. The $1/\sqrt{n}$ rate is the standard concentration rate for bounded random variables.

They also give improved bounds under Lipschitz continuity and monotonicity, showing the standard CRC rate as a special case. For distribution shift, they extend the result via importance weighting in the standard covariate-shift framework.

The key practical insight from the paper: approaches that account directly for finite-sample deviations across the grid achieve more stable risk control than approaches that first apply a monotonicity transformation to the loss and then run standard CRC. The transformation step introduces bias that the finite-sample analysis exposes.

---

## What this means for insurance model validation

The immediate application is expanding the set of loss functions you can use with CRC while retaining a theoretical guarantee.

Until this paper, the practical choice for insurance validation with CRC was either (a) restrict yourself to monotone losses that fit the standard framework, or (b) use a non-monotone loss but accept that you are relying on asymptotic arguments without finite-sample validity. Option (a) is limiting. Option (b) is defensible for large calibration sets but uncomfortable for specialist commercial lines books where $n$ might be 2,000–5,000.

The new rate O(√(log(m)/n)) tells you concretely what the guarantee costs. For a grid of $m = 100$ values and a calibration set of $n = 5{,}000$ policies:

$$\sqrt{\frac{\log 100}{5000}} = \sqrt{\frac{4.6}{5000}} \approx 0.030$$

That is approximately 3 percentage points of excess risk above target — which is the price of the finite-sample guarantee with non-monotone losses at this sample size. Whether that slack is acceptable depends on the specific application. For a model targeting 5% risk, a 3pp excess risk bound means the guarantee covers you up to 8% — still meaningful, and still a genuine finite-sample statement.

A second application is Winkler-score calibration. We have been asked several times why `insurance-conformal` does not include a Winkler-score CRC controller. The answer was that the theoretical foundation was not there. With this paper, it is. We will add `WinklerController` to `insurance-conformal.risk` once we have implemented and tested the non-monotone CRC calibration procedure — likely v1.4.0.

---

## Honest limitations

This is a theory paper. There are no insurance experiments, no Python code, and no empirical validation on claims data. The O(√(log(m)/n)) bound is the worst case over all bounded non-monotone losses — for specific losses with structure (e.g., unimodal risk as a function of $\lambda$), tighter problem-specific bounds may be available and the paper does not exploit this.

The grid size $m$ matters in the bound. For dense grids ($m = 1{,}000$), the $\sqrt{\log 1000} \approx 2.6$ factor versus $\sqrt{\log 100} \approx 2.1$ for a coarser grid is modest but not negligible. In practice, insurance applications tend to use coarse grids over economically meaningful loadings, so this is not a concern in most settings.

The paper also does not address the case where the risk function is continuous but non-monotone, and you want to optimise over a continuous $\lambda$. The finite-grid framework is natural for implementations but there will be problems where the grid approximation introduces meaningful discretisation error.

The preprint was submitted 2 April 2026. It has not been peer reviewed.

---

## What we take from it

Standard conformal risk control is already the right framework for insurance model validation when the loss is monotone — we laid out the case in detail in [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/03/13/insurance-conformal-risk/). This paper removes the restriction that was most likely to cause problems in practice.

The Winkler score case matters most immediately. Interval scoring rules are the standard way to evaluate probabilistic forecasts in meteorology, macroeconomics, and increasingly in actuarial science. Using CRC with the Winkler score loss gives you a prediction interval that is calibrated not just to coverage probability but to the expected value of the interval-scoring-rule loss — a materially stronger claim about the quality of your uncertainty estimates than "this interval contains the true value 95% of the time".

The two-sided regulatory test case has longer-term relevance. Solvency II Article 121 internal model requirements are well-established but validation practice continues to evolve. As validation frameworks mature, we expect two-sided performance criteria to become more common — "the model should neither significantly under-estimate nor significantly over-estimate" is a natural requirement that has no one-sided expression. Non-monotone CRC is the tool for that.

---

## The paper

Tareq Aldirawi, Yun Li and Wenge Guo, "Non-monotonicity in Conformal Risk Control", arXiv:2604.01502, April 2026. The main bound is Theorem 1; the lower bound establishing minimax optimality is Theorem 2; Lipschitz and monotone refinements are in Section 3; importance-weighted extension in Section 4.

---

## Related

- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/03/13/insurance-conformal-risk/) — conformal risk control foundations and `insurance-conformal.risk`
- [Conformal Prediction for Insurance Python: A Frequency-Severity Tutorial](/2026/04/04/conformal-prediction-insurance-python/) — the practical frequency-severity conformal model whose risk-control properties this paper extends
- [Conformal Prediction Works on Average — But Does It Work for Your Riskiest Customers?](/2026/04/04/conditional-validity-index-conformal-model-selection/) — conditional coverage and the CVI
- [Your Joint Prediction Sets Are 20–40% Too Wide](/2026/04/03/multivariate-conformal-prediction-mahalanobis-insurance/) — Mahalanobis non-conformity scores
