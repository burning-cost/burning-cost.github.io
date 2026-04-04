---
layout: post
title: "CRPS-Optimal Conformal Binning: When Actuarial Scoring Drives the Interval"
date: 2026-04-01
categories: [techniques, pricing]
tags: [conformal-prediction, CRPS, binning, prediction-intervals, insurance-conformal, CQR, LoBoostCP, heteroscedastic, dynamic-programming, Venn-prediction, scoring-rules, motor, severity, arXiv-2603-22000, Toccaceli, python]
description: "Paolo Toccaceli's CRPS-Optimal Binning for Conformal Regression (arXiv:2603.22000) partitions the covariate space using dynamic programming to minimise LOO-CRPS, then calibrates standard conformal prediction within each bin. On heteroscedastic data, it produces intervals 2.2× narrower than Gaussian split conformal while maintaining coverage. The CRPS calibration criterion directly aligns with how actuaries already evaluate probabilistic forecasts — which matters more than it sounds."
math: true
author: burning-cost
---

Every conformal prediction interval in our [`insurance-conformal`](/insurance-conformal/) library carries a marginal coverage guarantee: over repeated sampling, 90% of intervals will contain the true value. What none of them guarantee — unless you use CQR, LoBoostCP, or ShapeAdaptiveCP — is that the intervals are any good. A 90% interval that spans £0 to £50,000 for every motor policy is valid. It is also worthless.

Interval *efficiency* — getting the coverage you need with the narrowest possible intervals — is what determines whether conformal prediction is useful in practice. And efficiency depends entirely on how well the calibration procedure models the conditional distribution of residuals.

A paper from Paolo Toccaceli (arXiv:2603.22000, March 2026) takes a different angle on this problem than anything else in the conformal literature. Instead of asking "what quantile model should I fit?" (CQR) or "how should I localise within a gradient boosted tree?" (LoBoostCP), it asks: "what partition of the covariate space minimises the Continuous Ranked Probability Score?" The CRPS is the standard proper scoring rule for probabilistic forecasts in insurance. Using it to drive conformal calibration is a conceptually clean move.

---

## The binning idea

Split conformal prediction applies one calibration quantile to the whole dataset. Mondrian conformal prediction applies one calibration quantile per pre-defined group — rating cells, age bands, territory. The Mondrian approach achieves conditional coverage within each group but requires those groups to be pre-specified and to contain enough calibration data.

Toccaceli's approach is different: it finds the groups automatically, optimised specifically to minimise CRPS.

The setup is simple. Sort calibration observations by a covariate (or predicted value) $x_1 \leq x_2 \leq \cdots \leq x_n$. Partition them into $K$ contiguous bins. Within each bin, use the within-bin empirical CDF as the predictive distribution for new observations that fall in that bin. Apply standard conformal prediction using a CRPS-based nonconformity score. The bin boundaries and number of bins are chosen by dynamic programming to minimise leave-one-out CRPS.

The result is a non-parametric conditional density estimator — no quantile regression models, no assumed distribution family, no auxiliary predictions needed beyond the sort key. Just the empirical CDF of calibration residuals in the bin where a new observation lands.

---

## The maths

For a bin $S = \{y_1, \ldots, y_m\}$, the total leave-one-out CRPS decomposes to a closed form. The CRPS for a predictive CDF $\hat{F}$ and observed outcome $y$ is:

$$\text{CRPS}(\hat{F}, y) = \int_{-\infty}^{\infty} (\hat{F}(t) - \mathbf{1}[t \geq y])^2 \, dt$$

When $\hat{F}$ is the within-bin empirical CDF with one observation left out (leave-one-out), Toccaceli's Proposition 1 shows the total LOO-CRPS for a bin $S$ of size $m$ collapses to:

$$\text{cost}(S) = \frac{m}{(m-1)^2} \sum_{l < r} |y_l - y_r|$$

The sum of pairwise absolute differences $W = \sum_{l < r} |y_l - y_r|$ is the only quantity that matters. This is computable for any sub-interval of the sorted sequence using Fenwick trees in $O(n^2 \log n)$ time, producing a cost matrix $c(i, j)$ for every contiguous subsequence.

With that cost matrix precomputed, dynamic programming recovers the globally optimal $K$-partition in $O(n^2 K)$ time:

$$\text{dp}[k][j] = \min_{k-1 \leq i < j} \left\{ \text{dp}[k-1][i] + c(i+1, j) \right\}$$

No greedy heuristics. No local search. Globally optimal bin boundaries.

---

## The overfitting problem and how it is solved

Minimising LOO-CRPS within the calibration sample is biased. With $K$ large enough, you can make each bin contain exactly two observations with very similar $y$ values — the LOO-CRPS drops to near zero, but the bins have learned noise rather than structure. This is standard in-sample optimism.

The fix is elegant: use alternating-index cross-validation. Sort all observations by covariate value, then assign alternating indices to training and validation sets — odd indices to $\mathcal{T}$, even to $\mathcal{V}$. Fit the optimal $K$-partition on $\mathcal{T}$ using DP. Evaluate TestCRPS$(K)$ on $\mathcal{V}$. Repeat for $K = 1, \ldots, K_{\max}$ (typically $\lfloor n/10 \rfloor$).

The alternating assignment means training and validation are interleaved across the covariate space rather than split at a boundary — this preserves the covariate distribution in both halves. The result is a U-shaped TestCRPS curve with a genuine minimum at $K^*$.

---

## What CRPS buys you as the calibration criterion

CQR calibrates on pinball loss. LoBoostCP calibrates on absolute residuals within leaf-similar neighbourhoods. Both work, and we use both. But CRPS is a proper scoring rule — it is uniquely minimised when the predictive distribution matches the true conditional distribution. This matters.

Calibrating on pinball loss at two quantile levels optimises the interval endpoints, not the full distributional shape. Calibrating on absolute residuals (the standard nonconformity score) optimises point forecast accuracy, not distributional calibration. Calibrating on CRPS optimises the full predictive CDF — it penalises both miscalibration and excess width in the right way simultaneously.

For actuaries, this is not a novel metric. CRPS is what you use to score reserving distributions, catastrophe model outputs, and mortality forecasts. It is what Proper Scoring Rules in General Insurance (GIRO 2012) and the actuarial forecasting literature reach for when they want to evaluate full probability distributions, not just point forecasts. Building a conformal prediction method whose calibration criterion is CRPS is the right architectural choice. The scoring rule you optimise at calibration time and the scoring rule you use to evaluate the output are the same.

---

## The nonconformity score

Within a bin containing $m$ calibration observations $\{y_1, \ldots, y_m\}$, the CRPS-based nonconformity score for a candidate test outcome $y_h$ against the within-bin empirical CDF is:

$$\alpha(y_h) = \frac{1}{m} \sum_i |y_i - y_h| - \frac{W}{m^2}$$

where $W = \sum_{i < j} |y_i - y_j|$ is the within-bin pairwise dispersion. This score is convex in $y_h$, which guarantees the resulting prediction set $\Gamma^\varepsilon = \{y_h : p(y_h) > \varepsilon\}$ is always a single connected interval — not a union of disjoint sets. A key practical property: you always get an interval, never a disconnected set.

The conformal p-value at confidence level $\varepsilon$ gives a finite-sample marginal coverage guarantee:

$$P(y^* \in \Gamma^\varepsilon) \geq 1 - \varepsilon$$

under exchangeability of the calibration observations within the bin, which holds when the calibration set is an i.i.d. sample.

---

## The numbers: motorcycle and Old Faithful

The paper benchmarks on two real datasets against Gaussian split conformal, CQR with cubic spline quantile regression, and CQR-QRF (quantile random forest). Results are averaged over 200 random 50/50 train-calibration splits at nominal 90% coverage.

**Old Faithful (n=272):** Covariate is waiting time between eruptions; response is eruption duration. The conditional distribution is bimodal — short eruptions (~2 min) when waiting time is short, long eruptions (~4.5 min) when waiting time is long. Cross-validation selects $K^* = 2$ with one boundary at $x = 67.5$ minutes.

| Method | Coverage | Mean interval width |
|---|---|---|
| CRPS binning (full n) | 90.3% | 1.200 min |
| CQR-QRF | 91.4% | 1.333 min (+11%) |
| CQR cubic | 91.2% | 1.490 min (+24%) |
| Gaussian SCP | 91.2% | 1.683 min (+40%) |

The CRPS binning method achieves 40% narrower intervals than Gaussian split conformal while matching or exceeding the coverage of every competitor. The bimodal structure — completely inaccessible to a Gaussian assumption, hard to capture cleanly with quantile regression — is recovered by a single bin boundary.

**Motorcycle accident data (n=133):** Covariate is time post-impact; response is head acceleration. The variance is near-zero in the pre-impact phase and explosive during the 15–30ms deformation phase — classic heteroscedastic structure. Cross-validation selects $K^* = 10$ with bin boundaries clustered in the high-variance region.

| Method | Coverage | Mean interval width |
|---|---|---|
| CRPS binning (full n) | 91.0% | 78.9 g |
| CQR-QRF | 93.1% | 87.9 g (+11%) |
| CQR cubic | 92.5% | 134.1 g (+70%) |
| Gaussian SCP | 92.5% | 172.4 g (×2.2) |

Gaussian split conformal produces intervals 2.2 times wider than CRPS binning on the same data. CQR-QRF is competitive — only 11% wider — but requires a trained quantile forest.

---

## The insurance translation

These benchmark datasets are toy-scale, but the structures they contain are exactly what pricing models encounter.

The Old Faithful bimodal structure is the zero/non-zero claim problem. A motor policyholder either claims or does not. The conditional distribution of claim cost given the covariates is a mixture: a mass at zero (no claim) and a right-skewed distribution of positive values (claim occurred). Standard conformal prediction, calibrated on the full mixture, produces intervals that are too wide for zero-claim risks and too narrow for large-claim risks. CQR handles this partially — quantile regression at 5th and 95th percentile can distinguish the two modes if the model is well-specified. CRPS binning discovers the bimodal structure directly from the sorted residuals without requiring the model to know about it.

The motorcycle heteroscedastic structure is what you get in any reserve range or severity model where the variance of the outcome scales with the predicted value. A £50k large bodily injury reserve has higher absolute uncertainty than a £2k whiplash reserve. Gaussian SCP, applying a single calibration quantile to both, gives intervals that are grossly too wide for small claims and potentially too narrow for large ones. CRPS binning, by sorting on the predicted value and placing bin boundaries in the high-variance region, automatically allocates calibration data where variance is high.

The practical translation for a UK motor pricing team:

- **Frequency model:** Single sort key is the predicted frequency. Bin boundaries will cluster around the threshold between high-frequency (young urban driver) and low-frequency (mature rural driver) segments. CRPS-optimal bins do not need to know about age or urbanity — they find the partition that minimises calibration error, which will correlate strongly with your rating factors anyway.

- **Severity model:** Sort on predicted severity. Heteroscedasticity means bin boundaries concentrate in the high-severity region. The intervals for a £500 whiplash claim and a £30k fracture claim will be materially different widths, driven by empirical data rather than a parametric assumption.

- **Tweedie pure premium model:** Sort on $\hat\mu$ or $\hat\sigma$. The CRPS binning will discover the claim/no-claim structure without you encoding it explicitly.

---

## Where this fits in the insurance-conformal toolkit

We now have four approaches to interval calibration in [`insurance-conformal`](/insurance-conformal/), targeting different problems:

| Method | Conditional on | Requires auxiliary model | CRPS criterion |
|---|---|---|---|
| Split conformal (CQR) | Nothing — marginal only | Yes — quantile regression | No — pinball loss |
| LoBoostCP (v1.0.0) | Leaf similarity in GBT | No — uses existing GBT | No — absolute residuals |
| ShapeAdaptiveCP (MOPI) | Rating groups or RKHS | No — uses scale model | No — MSCE |
| **CRPS Binning** | Covariate rank order | No — uses sorted residuals | Yes |

The CRPS binning approach is the only one in the table that (a) requires no auxiliary model, and (b) optimises the full CRPS rather than a quantile or group-coverage objective. Its limitation is that it conditions on rank order in a single sort dimension — it is not multivariate. You sort by one covariate or score, find bin boundaries in that one dimension, and all the heterogeneity across other covariates is averaged within each bin. For a simple univariate problem or when you have a single good risk score, this is not a constraint. For a full GLM with ten rating factors, you need to decide what to sort on.

LoBoostCP's leaf-similarity approach naturally handles the multivariate case — the locality is defined in the full covariate space through the GBT's learned leaf assignments. For complex rating models on large portfolios, LoBoostCP (on CatBoost, XGBoost, LightGBM, or sklearn GBR) will generally be the better tool. For simpler models, smaller calibration sets, or any situation where you want the calibration criterion to be CRPS rather than absolute residuals, CRPS binning is worth trying.

ShapeAdaptiveCP (MOPI) is the right choice when conditional coverage by regulatory group is the primary concern — calibrating separately per age band, territory, or protected characteristic. CRPS binning does not target group coverage directly; it targets CRPS efficiency.

---

## Venn prediction bands

Alongside the standard conformal prediction sets, the method also produces Venn prediction bands. For a test point $x^*$ landing in bin $B_k$ with $m$ calibration observations, the Venn predictor outputs a band of CDFs:

$$\bar{F}(t) = \frac{\#\{i : y_i \leq t\} + 1}{m+1}, \quad \underline{F}(t) = \frac{\#\{i : y_i \leq t\}}{m+1}$$

The band has constant width $1/(m+1)$ everywhere, reflecting irreducible uncertainty about where $y^*$ sits within the bin's empirical distribution. As $m$ grows, the band narrows. This is the natural full-distributional output of the method — not just interval endpoints but a CDF band that can be integrated against any loss function, not just the $\pm$ coverage loss.

For reserving or capital model applications where you want to carry the full predictive distribution through downstream calculations, the Venn band is a principled representation. It is conservative by construction (the band always has positive width), but it is honest about what the data can support.

---

## Practical limitations

**Single sort key.** The method partitions on one-dimensional covariate ordering. In motor pricing, you might sort on predicted pure premium, predicted frequency, vehicle value, or driver age — but not on all of them simultaneously. Sorting on a composite risk score (e.g., the model's predicted value) is the natural choice and generally works well, but it commits you to treating the model's ordering as the right ordering for residual structure.

**Small calibration sets per bin.** The minimum bin size is $m_{\min} = 2$ (singleton bins have undefined LOO distributions). With $n = 500$ calibration observations and $K^* = 10$ bins, you have 50 observations per bin on average. Empirical quantile estimation from 50 observations has substantial sampling uncertainty — the finite-sample correction term in the conformal guarantee will be non-negligible. The alternating-split cross-validation helps select a $K^*$ that balances this, but for small datasets the method will favour fewer, wider bins.

**Not currently built.** `CRPSBinningCP` is not yet in `insurance-conformal`. Based on our assessment of the paper, this is a clean build — around 150-200 lines for the core class, with the main implementation effort in the Fenwick tree precomputation and DP backtracking. We are classifying it as a Phase 45 build candidate; the blog post is ahead of the implementation.

---

## Code sketch

The structural logic, before we build it properly:

```python
import numpy as np

def compute_w_matrix(y_sorted):
    """Pairwise dispersion W[i,j] = sum_{i<=l<r<=j} (y_r - y_l)
    for every contiguous sub-sequence of sorted y. O(n^2) time and space.
    (Production implementation uses Fenwick trees for O(n^2 log n).)
    """
    n = len(y_sorted)
    W = np.zeros((n, n))
    for i in range(n):
        running_sum = y_sorted[i]
        for j in range(i + 1, n):
            # Adding y[j] contributes (y[j] - y[l]) for each l in i..j-1.
            # Sum = (j-i)*y[j] - sum(y[i..j-1]) = (j-i)*y[j] - running_sum
            W[i, j] = W[i, j - 1] + (j - i) * y_sorted[j] - running_sum
            running_sum += y_sorted[j]
    return W

def cost(i, j, W):
    """LOO-CRPS cost for bin spanning indices i..j (0-indexed, inclusive)."""
    m = j - i + 1
    if m < 2:
        return np.inf
    return m / (m - 1) ** 2 * W[i, j]

def dp_optimal_partition(y_sorted, K):
    """Find K-partition minimising total LOO-CRPS via dynamic programming."""
    n = len(y_sorted)
    W = compute_w_matrix(y_sorted)
    INF = 1e18
    dp = np.full((K + 1, n + 1), INF)
    split = np.zeros((K + 1, n + 1), dtype=int)
    dp[0][0] = 0.0
    for k in range(1, K + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                c = dp[k - 1][i] + cost(i, j - 1, W)
                if c < dp[k][j]:
                    dp[k][j] = c
                    split[k][j] = i
    return dp[K][n], split

# Select K* via alternating-split cross-validation:
# sort by x, assign odd indices to T, even to V,
# fit DP on T for K = 1..K_max, evaluate TestCRPS on V, pick K* = argmin
```

This is $O(n^2)$ time for the W matrix and $O(n^2 K)$ for the DP — feasible for $n \leq 500$ in pure Python. The production version uses Fenwick trees for the W precomputation, bringing it to $O(n^2 \log n)$ and handling $n = 2{,}000$ calibration observations in a few seconds.

---

## The paper

Toccaceli, Paolo. "CRPS-Optimal Binning for Conformal Regression." arXiv:2603.22000 [stat.ML]. March 2026.

---

## Related posts

- [Shape-Adaptive Conformal Prediction: Why Your Intervals Are Wrong for Skewed Claims](/2026/04/01/shape-adaptive-conformal-prediction/) — ShapeAdaptiveCP (MOPI) for conditional coverage by group, with the masked-Z feature for GDPR-constrained UK pricing
- [Conformalised Quantile Regression: Prediction Intervals That Actually Adapt to Risk](/2026/03/24/conformalised-quantile-regression-insurance-prediction-intervals/) — CQR in insurance-conformal, including zero-inflation handling for motor claims
- [Two Ways to Control Risk in Automated Underwriting: Conditional vs Marginal](/2026/04/01/selective-conformal-prediction-automated-underwriting-conditional-vs-marginal-risk/) — SCRC and SCoRE for STP triage
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/03/13/insurance-conformal-risk/) — conformal risk control and the insurance-conformal library overview
