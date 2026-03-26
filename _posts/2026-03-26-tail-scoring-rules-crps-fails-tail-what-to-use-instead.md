---
layout: post
title: "Tail Scoring Rules: Why CRPS Fails in the Tail and What to Use Instead"
date: 2026-03-26
categories: [pricing, techniques, libraries]
tags: [evt, scoring-rules, severity, tail-index, hill-estimator, large-loss, pareto, gpd, insurance-severity, insurance-distributional, model-selection, crps, reinsurance, python]
description: "Brehmer & Strokorb (2019) proved that no proper scoring rule applied to raw data can discriminate tail indices. Bladt & Øhlenschlæger (arXiv:2603.24122) fix this by scoring normalised upper order statistics instead. Here is what it means for UK pricing and when it is actually usable."
---

If you are using CRPS to compare severity tail models, it is mathematically guaranteed to fail. Not "works poorly on small samples" or "tends to overfit" — guaranteed, provably, in the limit. Brehmer & Strokorb proved this in 2019 and it has largely gone unremarked in actuarial circles. Bladt & Øhlenschlæger (arXiv:2603.24122, March 2026) have a principled fix. It is theoretically elegant, honestly limited, and worth understanding now even if you cannot deploy it today.

---

## The impossibility result

When you compare two severity model fits using CRPS, log-score, or energy score — evaluated on the raw claims data — you are asking those scores to tell you which model better fits the tail. They cannot do this. Formally: for any two distributions F, G with different tail indices, you can construct predictive distributions whose scoring-rule values are arbitrarily close. Allen, Koh, Segers, and Ziegel (2025) extended this: you cannot fix it by taking the maximum over multiple scoring rules simultaneously.

This is not a problem with the scoring rules themselves. CRPS and log-score are perfectly proper — they correctly identify the true distribution in the limit, *for the full distribution*. The issue is that the tail contributes negligibly to the expected score when computed over the full data. For a UK motor bodily injury book where 95% of claims are under £20,000, the scoring-rule gradient from misspecifying behaviour at £500,000 is swamped by the gradient from the bulk of attritional claims. You can have the wrong Pareto tail index by a factor of two and the CRPS will not notice.

The practical consequence: every evaluation framework that rates severity models using a global proper scoring rule is blind to tail quality. AIC and BIC are equally blind — they are likelihood-based metrics that collapse across the full distribution. If you are pricing XL reinsurance or computing TVaR and you chose between a Pareto(0.5) and a Pareto(1.0) tail using CRPS, you may as well have flipped a coin.

---

## The fix: score normalised upper order statistics

Bladt & Øhlenschlæger's insight is to change what you score, not which score you use.

Sort your claims: Y_(1) ≤ Y_(2) ≤ ... ≤ Y_(n). Pick a threshold k (small relative to n). Take the top k observations, divide each by Y_(n−k) (the (k+1)-th largest, used as the threshold). You now have k normalised ratios, all in [1, ∞).

The theoretical fact that makes this work: under regular variation (Fréchet domain), those normalised ratios converge in distribution to i.i.d. draws from Pareto(1/γ_G), where γ_G is the true tail index. Once you are in the limit, the normalised upper order statistics no longer carry information about the bulk of the distribution. They only depend on the tail index. And any strictly proper scoring rule, applied to these normalised observations, will correctly identify γ_G.

The empirical tail log-score for candidate tail index γ is:

```
S_k(γ) = (1/k) Σ_{i=1}^k [ log(1/γ) − (1/γ + 1) · log(Y_(n−i+1) / Y_(n−k)) ]
```

Higher is better. You compare this across candidate values of γ (or candidate tail families), with a stability diagnostic over the range of k. The model whose γ tops the ranking at the smallest k values — where normalised extremes are most Pareto-like — is the preferred tail model.

---

## Hill estimation is secretly a scoring rule optimiser

The paper proves something clean in Theorem 6 and Corollary 8: the value of γ that maximises S_k under log-score is exactly the Hill estimator.

```
argmax_γ S_k(γ) = (1/k) Σ_{i=1}^k log(Y_(n−i+1) / Y_(n−k)) = Hill_k
```

This is not a coincidence. Hill (1975) was an MLE result; Segers (2001) showed it achieves minimum asymptotic variance among a broad class of residual estimators. What Bladt & Øhlenschlæger add is the scoring-rule interpretation: Hill is the score-optimal tail index estimator, and the tail score framework is its natural extension to *ranking competing models* on held-out data rather than estimating a single parameter.

The implication is practical: use Hill for estimating γ on your training data, and use the tail log-score framework for ranking competing model families on a holdout set. They are two sides of the same theoretical coin.

---

## What the Energy score does to heavy tails

The paper includes a table that should settle any temptation to use energy-score or CRPS variants for tail estimation (Table 1, n=10,000 Fréchet, k=0.05n, 100 replications):

| γ_G | Estimator | Bias (k=0.25n) |
|-----|-----------|----------------|
| 0.33 | Hill | 0.023 |
| 0.33 | Energy score | 0.021 |
| 0.66 | Hill | 0.048 |
| 0.66 | Energy score | 0.136 |
| 1.00 | Hill | 0.071 |
| 1.00 | Energy score | 0.712 |
| 1.33 | Hill | 0.095 |
| 1.33 | Energy score | 1.308 |

At γ = 1.33 — a tail index consistent with unlimited-compensation motor BI claims — the energy-score estimator has bias exceeding 1.0. The estimator is useless. For γ ≥ 1, CRPS and energy score require infinite expectations under the candidate distribution, so the finite-expectation constraint forces the scoring rule away from the true parameter. LogS does not have this problem: it remains valid for any γ > 0. For UK large-loss severity, where infinite-mean distributions (γ ≥ 1) are plausible, LogS is the only safe choice.

---

## The Python implementation

The core function is six lines of NumPy:

```python
import numpy as np

def tail_log_score(y: np.ndarray, gamma: float, k: int) -> float:
    """
    Empirical tail log-score S_k(Pareto(gamma)), eq. (4.1) of arXiv:2603.24122.
    Higher is better tail fit.
    y: observations (unsorted, all positive)
    gamma: candidate tail index (> 0)
    k: number of upper order statistics (1 <= k < len(y))
    """
    y_sorted = np.sort(y)
    threshold = y_sorted[-(k + 1)]               # Y_(n,n-k)
    top_k = y_sorted[-k:] / threshold            # normalised ratios in [1, ∞)
    log_density = np.log(1.0 / gamma) - (1.0 / gamma + 1.0) * np.log(top_k)
    return float(np.mean(log_density))
```

Asymptotic CIs follow from Corollary 4: the variance of the log-score under LogS is (1/γ − 1)² · γ_G², where γ_G is estimated by Hill. Substitute the Hill estimate, divide by k, take the square root, and you have a standard error that supports 95% pointwise CIs around each curve:

```python
def tail_score_ci(score: float, gamma: float, gamma_hill: float,
                  k: int, level: float = 0.95) -> tuple:
    from scipy.stats import norm
    var = (1.0 / gamma - 1.0) ** 2 * gamma_hill ** 2
    se = np.sqrt(var / k)
    z = norm.ppf(0.5 + level / 2.0)
    return (score - z * se, score + z * se)
```

To rank a set of candidate models, compute `tail_log_score` for each across a grid of k values and look for a stable region where the ranking is consistent. The k at which rankings stabilise is your evidence that you are in the regular variation regime.

---

## What it shows on the USAutoBI data

The empirical section uses 1,340 US automobile bodily injury claims from 2002 (CASdatasets R package). A Pareto QQ plot confirms a roughly linear upper tail. Five candidates: γ ∈ {0.3, 0.5, 0.8, 1.0, 1.3}.

Full sample: γ = 0.8 and γ = 1.0 jointly top the ranking over a stable k range. But the 95% CIs overlap across γ ∈ {0.5, 0.8, 1.0, 1.3}. With 1,340 claims, the method is suggestive, not decisive.

The more instructive result comes from the attorney subgroup. Split by whether the claimant had legal representation: with-attorney (n=685) looks like γ ≈ 0.8–1.0, consistent with the full sample. Without-attorney (n=655): γ = 0.5 tops the ranking, with γ = 0.3 competitive. Smaller, simpler claims avoid solicitors; the no-attorney subgroup has a materially lighter tail. The framework detects this heterogeneity where a full-distribution scoring rule would not.

No meaningful tail difference by sex — both subgroups maintain the full-sample ranking.

This is the framework's natural use case: detecting tail heterogeneity across subgroups, not just estimating a single tail index.

---

## Where it fits in the UK pricing workflow

The standard UK large-loss workflow runs something like: collect ground-up losses above reporting threshold, fit GPD / lognormal / spliced, compare by AIC and QQ plots, select model, price XL layer.

AIC and QQ plots are bulk-distribution diagnostics. They cannot tell you which model fits the tail better on held-out data, and the impossibility result confirms that CRPS cannot either. The tail log-score fills that gap — as a holdout evaluation metric, not a fitting criterion. Train on 70%, evaluate tail score on 30%, report CIs. If one model's tail score dominates another's outside the CI, that is your first principled evidence about tail quality.

The natural integration point is [`insurance-severity`](https://github.com/burning-cost/insurance-severity): after fitting `CensoredHill`, `TruncatedGPD`, or `WeibullTemperedPareto` to large-loss data, the tail log-score provides a model-selection step where currently only MLE and AIC exist. The tail index parameter from any of those fits is the γ you pass to `tail_log_score`.

For [`insurance-distributional`](https://github.com/burning-cost/insurance-distributional), the gap is in `scoring.py`. The library correctly implements CRPS, Tweedie deviance, Gini and CDE loss for full-distribution evaluation. A `tail_log_score` function accepting a `DistributionalPrediction` would complete the picture for severity heads — but requires the prediction object to expose a tail-index parameter or tail-limit distribution, which the current interface does not do. That is a plausible addition, not a trivial one.

If you are using the spliced lognormal-Pareto approach covered in [Spliced Severity Distributions]({{ site.baseurl }}{% post_url 2025-03-15-spliced-severity-distributions-when-one-distribution-isnt-enough %}), the tail log-score applies directly to the Pareto component above the splicing point — consistent with how those models are fitted. You would evaluate tail scores on the subset of claims above the splicing threshold, which requires a clean separation of the data.

---

## Honest limitations

This is a watch-and-note technique, not something to deploy next month. The restrictions are genuine and not surmountable by clever engineering.

**Fréchet domain only.** The framework requires regular variation: the survival function must behave like a power law in the tail. That means GPD with ξ > 0, Pareto, Burr. It explicitly excludes lognormal, Weibull, and Gamma — which are in the Gumbel domain (ξ = 0). UK motor severity below, say, £50,000 is typically lognormal-like. If your Pareto QQ plot is concave in the upper tail, you are not in the Fréchet domain and the tail score has no valid asymptotic justification. The Allen et al. (2025) tail calibration framework works across all three extreme value domains and is arguably more relevant for general UK use — that is a separate paper we intend to cover.

**Sample size.** The simulations are unambiguous: n ≥ 100,000 for reliable discrimination; n ~ 10,000 for moderate results; n ~ 1,000 for poor results (especially for Burr-type DGPs whose normalised extremes converge slowly to the Pareto limit). UK excess-of-loss reinsurance experience above £250,000 might yield 500–2,000 ground-up losses over a decade. The USAutoBI application with 1,340 claims had overlapping CIs across most candidate tail indices — that is an honest representation of what the method can do at typical UK large-loss sample sizes. You will get ranking information, not statistical decisions.

**k selection is manual.** The paper recommends finding a "stability range" where tail rankings are consistent as k varies. There is no automatic k-selection rule. In practice, you plot the tail score for each candidate γ across a grid of k and look for a plateau. That is a visual process, not a production-ready algorithm. Automating it requires a rank-variance heuristic that the paper does not provide.

**No censoring or truncation.** XL structures create censored large-loss data. Policy limits create truncation. The tail score framework has no mechanism for either. `TruncatedGPD` and `CensoredHill` in `insurance-severity` handle these, but integrating them with the tail scoring framework requires EVT theory for censored order statistics. That theory exists but is not in this paper.

**Pareto-only candidates.** The empirical section compares Pareto(γ) models with five different γ values. The real practitioner decision — GPD vs lognormal vs Burr vs spliced — is not demonstrated. The framework can in principle handle any Fréchet-domain family (their tail limits are Pareto-type), but the paper does not show this working.

---

## What to do now

Read the paper: [arXiv:2603.24122](https://arxiv.org/abs/2603.24122). The theoretical section is tight and the proofs are accessible if you are comfortable with regular variation.

Run the `tail_log_score` function above on your own large-loss data. Plot the scores across a k grid for a handful of candidate GPD tail indices. You will immediately see whether your data has enough signal to discriminate — if the CI bands overlap everywhere, you are in the n-too-small regime and need to acknowledge that.

Do not drop AIC and QQ plots. They evaluate the full distribution; the tail score evaluates only the tail. You need both, and a model that wins on tail score while losing badly on AIC probably has a bulk-fit problem that will hurt frequency estimates.

We will add a `tail_log_score` module to `insurance-severity` once the k-selection stability logic and Fréchet domain diagnostic are solid enough to be useful rather than misleading. The core function is already in the KB implementation sketch (entry 4027). The surrounding guardrails — Pareto QQ test, Hill plot, CI computation — are the parts that need thought before they go near production code.

The fundamental finding is worth carrying regardless: if your severity model evaluation uses only bulk-distribution metrics, you have no principled basis for your tail model choices. That is not a gap you want in an XL pricing or TVaR calculation.
