---
layout: post
title: "Measuring Proxy Discrimination in Insurance Pricing: The LRTW Scalar Metric"
date: 2026-03-31
categories: [fairness]
tags: [proxy-discrimination, FCA, Consumer-Duty, LRTW, sensitivity-analysis, Sobol, Shapley, constrained-regression, insurance-fairness-diag, Equality-Act, python]
description: "Lindholm, Richman, Tsanakas and Wüthrich (EJOR, January 2026) give us a scalar proxy discrimination measure with a property none of the existing methods share: PD=0 if and only if the price is genuinely free from proxy discrimination."
math: true
---

We have [covered proxy discrimination detection before](/2026/03/22/three-methods-proxy-discrimination-detection-compared/). We have covered the regulatory exposure. We have covered the three-method interpretation problem — when mutual information, proxy R-squared, and SHAP proxy scores disagree, which one to trust.

This post covers something different: why all of those methods are insufficient in the same way, and what a rigorous scalar measure actually looks like. The source is Lindholm, Richman, Tsanakas and Wüthrich, published in the *European Journal of Operational Research* in January 2026 (open access, [City Research Online](https://openaccess.city.ac.uk/id/eprint/36642/)). The mathematical framework they introduce is the strongest academic answer to the regulatory question of how to quantify proxy discrimination in insurance.

The short version: they define a metric, PD, with a tight characterisation — PD = 0 if and only if the pricing model avoids proxy discrimination. No existing method has this property. Understanding why requires understanding what the existing methods are actually measuring.

---

## What the existing approaches miss

The obvious first step in proxy discrimination testing is correlation. Run a regression of each rating factor on the protected characteristic (ethnicity, say), compute the R-squared. If it is high, flag the factor. This is what most pricing teams actually do.

The problem is that correlation measures the association between a factor and the protected attribute. It says nothing about whether that association is transmitted into prices. A factor can correlate strongly with ethnicity and contribute almost nothing to price variance — it is technically a proxy but is causing no material disadvantage. The Equality Act Section 19 test requires disadvantage, which requires price impact. Correlation gives you neither.

The next step up is demographic unfairness — the proportion of price variance attributable to the protected attribute D. Define it formally as:

$$\text{UF}(\pi) = \frac{\text{Var}(\mathbb{E}[\pi(X) \mid D])}{\text{Var}(\pi(X))}$$

This is a first-order Sobol index. It measures how much of the spread in premiums can be explained by D alone. Unlike correlation, it directly concerns prices rather than factors. It is scale-invariant: adding a loading or multiplying by a constant does not change UF.

But UF has a critical limitation. UF = 0 means that average premiums are equal across D groups. It does not mean that the premium distribution is the same across D groups. Two distributions can have identical means and completely different shapes. UF = 0 is demographic parity in mean only. A model that is systematically overcharging high-risk individuals within each D group — where risk correlates with D even after conditioning on X — can produce UF = 0 while still proxy-discriminating in the distributional sense.

More fundamentally: UF = 0 does not imply that π(X) is statistically independent of D. UF checks one moment of one conditional distribution. Independence requires all moments of all conditional distributions to match. These are very different conditions.

The LRTW paper proves this explicitly (Proposition 1). UF = 0 implies demographic parity in mean, not discrimination-free pricing. The two are not the same.

---

## The admissible price set

To define a tight measure of proxy discrimination, LRTW first define what it means to avoid it. The notation: Y is the response (claim frequency or loss), X is the vector of legitimate rating factors, D is the protected attribute taking values in a finite set.

The best-estimate price is:

$$\mu(x, d) = \mathbb{E}[Y \mid X = x, D = d]$$

This uses D directly — it is direct discrimination, clearly impermissible. The standard response is unawareness: remove D from the model entirely. The unawareness price is:

$$\mu(x) = \mathbb{E}[Y \mid X = x]$$

But here is the mechanism of proxy discrimination. The unawareness price satisfies:

$$\mu(x) = \sum_{d \in \mathcal{D}} \mu(x, d) \cdot \mathbb{P}(D = d \mid X = x)$$

The weights on the right — $\mathbb{P}(D = d \mid X = x)$ — depend on x. So the unawareness price is, effectively, a weighted average of the group-specific best-estimate prices, where the weights vary by covariate profile. The model has never seen D. It does not need to. The weights are implicit in the joint distribution of X and D in the training data.

A price is *admissible* — free from proxy discrimination — if it can be written as:

$$\pi(x) = c + \sum_{d \in \mathcal{D}} \mu(x, d) \cdot v_d$$

for some constant c and weights v that do **not** depend on x. The weights must satisfy $0 \leq v_d \leq 1$ and $\sum_d v_d \leq 1$. The x-independence of the weights is the key condition: the model can use different risk levels by group (each $\mu(x,d)$ varies with x), but it cannot vary the weighting across covariate profiles. Varying the weights by x is exactly how proxy discrimination enters.

The discrimination-free price from Lindholm et al. (2022) — $h^*(x) = \sum_d \mu(x,d) \cdot \mathbb{P}^*(D = d)$ with a reference distribution over D — is a special case of this with $c = 0$ and $v_d = \mathbb{P}^*(D = d)$.

---

## The PD metric

Now we can define proxy discrimination properly. The metric is the normalised squared distance from the fitted price to the nearest admissible price:

$$\text{PD}(\pi) = \min_{c \in \mathbb{R},\, v \in V} \frac{\mathbb{E}\!\left[\left(\pi(X) - c - \sum_{d} \mu(X, d)\, v_d\right)^2\right]}{\text{Var}(\pi(X))}$$

After optimising out the constant c, this becomes the residual variance from regressing $\pi(X)$ on $\{\mu(X, d) : d \in \mathcal{D}\}$, with the constraint that the regression coefficients lie in the simplex-like set V, divided by $\text{Var}(\pi(X))$.

The key property (Proposition 2): **PD = 0 if and only if π avoids proxy discrimination.** This is what makes PD different from everything else. It is a tight characterisation in both directions. Not just "PD = 0 is a necessary condition" or "a sufficient condition" — it is both. No other widely-used fairness metric in insurance has this property.

The other properties: PD lies in $[0, 1]$, it is scale-invariant under positive affine transformations, and PD = 1 when the premium is uncorrelated with all group-specific best-estimate prices — maximum proxy discrimination.

Computationally, PD requires solving a constrained quadratic programme. The number of variables equals $|\mathcal{D}|$ — the number of distinct values of the protected attribute. For binary protected attributes (e.g., gender) this is a two-variable QP. For ethnicity coded to five groups, it is five variables. This is a small problem: scipy's SLSQP solver handles it in milliseconds.

```python
import numpy as np
from scipy.optimize import minimize

def compute_pd(pi: np.ndarray, mu_by_d: dict, weights: np.ndarray = None) -> dict:
    """
    Compute LRTW 2026 proxy discrimination metric (Definition 4, Eq. 7).

    Parameters
    ----------
    pi : array (n,)
        Fitted prices from the unawareness model.
    mu_by_d : dict {d_value: array (n,)}
        Best-estimate prices mu(X, d) for each value of D.
        Requires all group-specific models to be estimated.
    weights : array (n,), optional
        Exposure weights. If None, uniform weighting.

    Returns
    -------
    dict with keys: pd_metric, v_star, c_star, lambda_ (discrimination residual)
    """
    if weights is None:
        weights = np.ones(len(pi))
    w = weights / weights.sum()

    d_values = list(mu_by_d.keys())
    mu_matrix = np.column_stack([mu_by_d[d] for d in d_values])  # (n, |D|)

    def var_w(x):
        mean_x = np.dot(w, x)
        return np.dot(w, (x - mean_x) ** 2)

    var_pi = var_w(pi)

    def objective(params):
        c = params[0]
        v = params[1:]
        admissible = c + mu_matrix @ v
        residual = pi - admissible
        return np.dot(w, residual ** 2)

    n_d = len(d_values)
    x0 = np.zeros(1 + n_d)
    x0[0] = np.dot(w, pi)  # initialise c at weighted mean

    # Constraints: 0 <= v_d <= 1 for all d, sum(v) <= 1
    bounds = [(-np.inf, np.inf)] + [(0.0, 1.0)] * n_d
    constraints = {"type": "ineq", "fun": lambda p: 1.0 - p[1:].sum()}

    result = minimize(objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 2000})

    c_star = result.x[0]
    v_star = result.x[1:]
    lambda_ = pi - c_star - mu_matrix @ v_star  # discrimination residual

    pd_metric = np.dot(w, lambda_ ** 2) / var_pi

    return {
        "pd_metric": pd_metric,
        "v_star": dict(zip(d_values, v_star)),
        "c_star": c_star,
        "lambda_": lambda_,           # per-policyholder delta_PD scores
        "admissible_price": pi - lambda_,
    }
```

The `lambda_` output is the per-policyholder proxy discrimination exposure score: $\delta_\text{PD}(x) = \Lambda(x)$. Positive means the policyholder is overpriced relative to the nearest discrimination-free benchmark. Crucially, it is a function of X only — you do not need D to be observed for that individual to compute it.

---

## Attributing PD to rating factors

A scalar metric is a verdict. Attribution tells you what to fix. LRTW provide attribution via the Owen (2014) Shapley decomposition of variance — distinct from SHAP (Lundberg and Lee, 2017), which decomposes individual predictions. The Owen framework decomposes a global variance metric. The value function is:

$$w(S) = \text{Var}\!\left(\mathbb{E}[\Lambda(\pi, X) \mid X_S]\right)$$

The Shapley attribution for feature i is:

$$\text{PD}^\text{sh}_i(\pi) = \frac{1}{\text{Var}(\pi(X))} \cdot \frac{1}{q} \sum_{S \subseteq \{1,\ldots,q\} \setminus \{i\}} \binom{q-1}{|S|}^{-1} \left[w(S \cup \{i\}) - w(S)\right]$$

Two properties matter here. First, additivity: $\sum_i \text{PD}^\text{sh}_i = \text{PD}(\pi)$. The attribution sums to the global metric. "Postcode accounts for 37% of proxy discrimination" is directly interpretable because the denominator is $\text{Var}(\pi(X))$ throughout, not $\text{Var}(\Lambda)$. Second, some $\text{PD}^\text{sh}_i$ can be negative. A factor that absorbs or mediates the D–Y relationship — reducing other factors' ability to proxy D — gets a negative attribution. This is a diagnostic finding: it identifies factors that are currently doing beneficial work in the price structure.

```python
import numpy as np
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor

def shapley_proxy_attribution(
    lambda_: np.ndarray,
    X: np.ndarray,
    var_pi: float,
    feature_names: list[str],
    n_permutations: int = 512,
) -> dict[str, float]:
    """
    Owen 2014 Shapley decomposition of PD (Definition 6, Eq. 12).

    Estimates w(S) = Var(E[Lambda | X_S]) for sampled coalitions S by fitting
    an RF surrogate on X_S and computing variance of in-sample predictions.
    Coalition results are cached — each unique subset is fitted only once.

    For q > 12, prefer TreeSHAP with interventional=False on the RF surrogate;
    see the text below for why.

    Note: CEN-SHAP (Richman & Wüthrich 2023, SSRN 4514891) is the exact method
    used in the LRTW paper. No Python port exists as of March 2026.
    """
    q = X.shape[1]
    coalition_cache: dict[tuple, float] = {}

    def coalition_variance(features_idx: tuple[int, ...]) -> float:
        """Estimate Var(E[Lambda | X_S]) via RF surrogate, with caching."""
        if features_idx in coalition_cache:
            return coalition_cache[features_idx]
        if len(features_idx) == 0:
            coalition_cache[features_idx] = 0.0
            return 0.0
        X_sub = X[:, list(features_idx)]
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_sub, lambda_)
        e_hat = rf.predict(X_sub)
        result = float(np.var(e_hat, ddof=0))
        coalition_cache[features_idx] = result
        return result

    rng = np.random.default_rng(0)
    shapley_effects: dict[str, float] = {}

    for i, name in enumerate(feature_names):
        others = [j for j in range(q) if j != i]
        marginal_contributions = []

        for _ in range(n_permutations):
            # Sample a random ordering of the other features
            perm = rng.permutation(others).tolist()
            # Pick a random split point — equivalent to a random coalition S
            split = rng.integers(0, len(perm) + 1)
            s_without = tuple(sorted(perm[:split]))
            s_with = tuple(sorted(perm[:split] + [i]))
            marginal_contributions.append(
                coalition_variance(s_with) - coalition_variance(s_without)
            )

        shapley_effects[name] = float(np.mean(marginal_contributions)) / var_pi

    return shapley_effects
```

The coalition cache is load-bearing here. Without it, the inner loop would refit a new RF model for every permutation sample — O(n_permutations × q) model fits per feature, times q features. With it, each unique coalition is fitted once and reused across all features and permutations that happen to sample the same subset. For q ≤ 12, the full set of $2^q$ coalitions fits comfortably in memory.

For q > 12 — which covers most UK personal lines rating structures — the exact Owen estimator is infeasible even with caching. The pragmatic alternative is TreeSHAP with `interventional=False` (conditional SHAP) on the RF surrogate: not identical to Owen 2014 but computationally tractable and a close approximation when the surrogate fits well. The key distinction from standard SHAP: conditional SHAP respects feature correlations, which matters for rating factors where postcode correlates with nearly everything else.

The LRTW paper uses Conditional Expectation Networks (Richman and Wüthrich, SSRN 4514891) as the surrogate. CEN trains a single neural network that takes $(X, \text{mask})$ as input, where mask is a binary vector indicating which features are observed — enabling all $2^q$ conditional expectations from a single model fit. CEN is only available in R as of March 2026; there is no Python port.

---

## What the case study shows

The paper applies PD to a portfolio of 165,511 policies. The results are instructive. The unawareness price gives PD = 0.00277 (SE = 0.0000203) and UF = 0.0639 (SE = 0.00079). These two numbers appear to be measuring similar things at different scales, but they are not comparable: UF measures how much of premium variance is explained by D-group membership on average; PD measures how far the price structure is from the admissible set. The same UF could arise from a model that is slightly off across every covariate, or from one that is severely off in a small subset.

The Shapley attribution (Table 2 in the paper) identifies a geography-type covariate — essentially a postcode-like feature — as the dominant contributor to PD. This is consistent with what we found running correlation-based proxy detection on UK motor books. Postcode is not just anecdotally problematic; it consistently dominates when you compute the attribution properly.

The paper also shows something the correlation approach misses entirely: some rating factors have **negative** Shapley effects on PD. These factors are not contributing to proxy discrimination — they are attenuating it, absorbing variation in Lambda that would otherwise be attributed elsewhere. Identifying these factors matters for remediation: removing them might actually increase PD.

---

## Why the existing library falls short

[`insurance-fairness-diag`](/insurance-fairness-diag/) implements a D_proxy metric defined as $\sqrt{\text{UF}}$ — the square root of the demographic unfairness Sobol index. That is not PD. The distinction matters.

The group-mean approximation in the current `_admissible.py` finds the closest admissible price by projecting each observation onto the mean premium for its observed D group. The full PD metric finds the closest point in the admissible set by regressing $\pi(X)$ on the complete set of group-specific best-estimate prices $\{\mu(X, d) : d \in \mathcal{D}\}$ with x-independent weights. These are the same computation only when X and D are independent — which, for any rating factor that is a proxy, they are not.

Put concretely: if postcode area is a proxy for ethnicity, $\mathbb{P}(D = d \mid X = x)$ varies across postcode areas. The group-mean approximation cannot separate the legitimate risk component of that variation from the discriminatory component. The constrained QP can, because it explicitly looks for the best x-independent weighting of the group-specific models.

The gap is the constrained QP in `compute_pd()` above. The Shapley surrogate approach is already in `_shapley.py`; it just needs Lambda as input rather than the simplified residual. Unit-testing against Example 3 from the paper — where D ∈ {0,1}, X ∼ U(0,1), $\mathbb{P}(D=1|X) = X$, and the analytical solution is PD = 0.25 — confirms the implementation is correct before touching production data.

---

## The regulatory angle

The FCA's December 2024 research note on bias in supervised machine learning ([FCA, 2024](https://www.fca.org.uk/publications/research-notes/research-note-literature-review-bias-supervised-machine-learning)) identifies, without prescribing a specific methodology, exactly the capabilities that PD and its Shapley decomposition provide:

- "A feature which causes bias must be associated with a demographic characteristic... but this is not sufficient for contributing to bias as it must also be included in and important to a model's predictions." PD separates correlation (X–D association) from price transmission via the constrained regression.
- "Identify which variables most influence the racial bias of a credit risk model." That is the Shapley decomposition.
- "Location for motor insurance" is explicitly flagged as a common proxy variable — consistent with the LRTW case study finding.

The FCA research note adds the standard caveat that it "may not necessarily represent the position of the FCA." There is no prescribed methodology, no materiality threshold, and no requirement to implement any specific framework as of March 2026. The regulatory demand is currently best read as: demonstrate fair value at segment level with evidence. The PD metric and per-policyholder Lambda scores are strong evidence. They connect to Consumer Duty (PRIN 2A.3) through the fair value obligation: not just "average premiums differ across groups" but "these specific policies are systematically overpriced by £X relative to the proxy-discrimination-free benchmark."

The correct legal hook is Equality Act 2010 Section 19 (indirect discrimination) combined with Consumer Duty fair value obligations — not FCA EP25/2, which evaluates GIPP price-walking remedies and is a separate matter entirely.

---

## Honest limitations

Several limitations matter for production use.

**D must be observed.** PD requires estimating $\mu(x, d)$ for all values of d in the training data. If D is not recorded — which is common for ethnicity in UK insurance — you cannot compute PD directly. Proxy inference (inferring likely D from postcode demographics) adds model uncertainty and the FCA's own research note calls it "fraught with the potential for inaccurate guesswork." The Zhang, Liu and Shi (2025) approach using local differential privacy to privatise sensitive attributes during training is a partial answer but is not yet production-ready.

**Shapley complexity.** For q = 15–25 rating factors, exact Owen 2014 is infeasible even with caching. The RF surrogate with random permutation sampling is a practical approximation, but the Monte Carlo noise is material for small portfolios. Bootstrap confidence intervals on the Shapley effects require running the full computation repeatedly — expensive.

**No materiality threshold.** PD = 0.003 from the LRTW case study sounds small. Whether it represents material disadvantage under Consumer Duty requires translating it into monetary terms: the mean absolute value of Lambda in pounds. The paper provides guidance on this in the supplementary materials but prescribes no universal threshold.

**No causal interpretation.** High PD means the price structure is far from the admissible set. It does not mean the model is behaving maliciously, or even incorrectly given available risk data. A postcode effect that genuinely reflects claims risk in a particular area will also contribute to PD if that area is demographically distinctive. The actuarial justification — demonstrating that the X–D correlation has a risk-based explanation — is separate work.

**L2 loss.** The constrained QP minimises squared Euclidean distance. For frequency-severity models with Tweedie or Gamma structure, L2 loss on the linear predictor may not be the natural metric. The paper does not address extensions to non-Gaussian loss functions.

---

## What to build next

The constrained QP for PD is a small addition to the existing `insurance-fairness-diag` stack. The existing Owen 2014 permutation estimator in `_shapley.py` is reusable with Lambda as the target. The Python surrogate workaround is sufficient for portfolios with up to 20 rating factors if you accept the approximation.

The larger gap is CEN: the Conditional Expectation Network that enables exact Shapley computation over all $2^q$ coalitions from a single model fit. Until a Python port exists, the RF surrogate approximation is what we have. For most practical purposes — a pricing team needing a defensible number for an FCA file review — the approximation is adequate. The exact result matters for research papers; it matters less for internal audit documentation.

The unit tests are the priority. PD = 0.25 on Example 3, PD = 0.00277 on the case study. If you cannot pass those, the number you report means nothing.

---

## Reference

Lindholm, M., Richman, R., Tsanakas, A. and Wüthrich, M.V. (2026). Sensitivity-based measures of discrimination in insurance pricing. *European Journal of Operational Research*. DOI: [10.1016/j.ejor.2026.01.021](https://doi.org/10.1016/j.ejor.2026.01.021). Open access: [City Research Online](https://openaccess.city.ac.uk/id/eprint/36642/).
