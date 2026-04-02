---
layout: post
title: "Sequential Optimal Transport for Multi-Attribute Fairness in Insurance Pricing"
date: 2026-03-31
categories: [fairness, machine-learning]
tags: [optimal-transport, fairness, demographic-parity, Wasserstein-barycenter, sequential-correction, insurance-fairness, Equality-Act-2010, FCA-Consumer-Duty, Lindholm-2022, Hu-Ratz-Charpentier-2024, multi-attribute, protected-characteristics, insurance-pricing]
description: "Naively applying Wasserstein barycenter corrections sequentially across multiple protected attributes is miscalibrated: the ECDF for attribute k was fitted on the original predictions but applied to already-corrected ones. We explain the bug, the Hu-Ratz-Charpentier (AAAI 2024) fix, and the correct pipeline for UK insurance — where demographic parity is not the regulatory standard."
author: burning-cost
---

Fairness corrections in insurance pricing almost always arrive as single-attribute problems. Correct for gender. Correct for age. Each correction gets implemented, validated, and signed off in isolation. The difficulty begins when you need to correct for two or three protected attributes simultaneously — which is the realistic case for any UK insurer operating under the Equality Act and FCA Consumer Duty simultaneously.

The tempting implementation is to apply a Wasserstein barycenter correction for gender, calibrate a second corrector for ethnicity proxies on the original predictions, then chain them. This is miscalibrated in a specific, fixable way. This post explains the bug, the correct algorithm from Hu, Ratz & Charpentier (AAAI 2024, arXiv:2309.06627), and where OT correction fits in a UK pricing fairness pipeline.

---

## The single-attribute OT correction

Start with the single-attribute case, which is correct and well-understood.

You have a fitted pricing model that produces predictions $f^*(x, a)$ where $x$ is the risk feature vector and $a$ is a protected attribute (say, gender with groups $\{0, 1\}$). The Wasserstein barycenter correction finds the distribution that is the "midpoint" between the two group-conditional prediction distributions, in the Wasserstein-2 sense, and maps each group's predictions onto it.

Formally, for a protected attribute $A$ with groups $a \in \mathcal{A}$:

1. Compute the exposure-weighted ECDF $F_a$ and its inverse (the empirical quantile function, EQF) $Q_a$ for each group
2. Compute the barycenter quantile function:

$$\bar{Q}(u) = \sum_{a \in \mathcal{A}} \omega_a \, Q_a(u)$$

where $\omega_a$ is the portfolio exposure weight for group $a$

3. The OT map for group $a$ is $T_a(z) = \bar{Q}(F_a(z))$
4. The fair prediction for policy $i$ with group $a_i$ is:

$$f_B(x_i) = T_{a_i}(f^*(x_i)) = \bar{Q}(F_{a_i}(f^*(x_i)))$$

The resulting predictor achieves Strong Demographic Parity: $f_B \perp A$. The distribution of predictions is identical across groups. The cost of this correction, measured in Wasserstein-2 distance, is minimised by the barycenter construction — you are moving each group's distribution the smallest possible distance to a common target.

The $\varepsilon$-partial correction provides an explicit accuracy-fairness trade-off:

$$f_\varepsilon = (1 - \varepsilon) \cdot f_B + \varepsilon \cdot f^*$$

At $\varepsilon = 0$ you have full demographic parity. At $\varepsilon = 1$ you have the uncorrected model. Values in between give a continuum of trade-offs. This is the right lever for a UK insurer who wants to demonstrate directional fairness improvement without fully homogenising the risk signal.

---

## The multi-attribute calibration bug

Now suppose you have two protected attributes: $A_1$ (e.g. gender proxy) and $A_2$ (e.g. ethnicity proxy via postcode cluster). The naive extension is to run the single-attribute correction twice:

1. Fit $F_{a_1}$ and $\bar{Q}_1$ on $f^*(x_i)$, the original predictions
2. Apply to get $f_1(x_i) = \bar{Q}_1(F_{a_1}(f^*(x_i)))$
3. Fit $F_{a_2}$ and $\bar{Q}_2$ on $f^*(x_i)$, the original predictions again
4. Apply to get $f_2(x_i) = \bar{Q}_2(F_{a_2}(f_1(x_i)))$

Step 4 is where it breaks. You are applying $F_{a_2}$ — which was calibrated on the distribution of $f^*$ — to values drawn from the distribution of $f_1$. These are different distributions. The ECDF $F_{a_2}$ is miscalibrated: it maps $f_1$ values to probability levels as if they were $f^*$ values. The quantile mapping $\bar{Q}_2(F_{a_2}(\cdot))$ is no longer a valid OT transport from $f_1$ to the $A_2$-barycenter.

In practice this means the $A_2$ correction is neither correctly correcting for $A_2$ nor preserving the $A_1$ correction. The two corrections interfere. You can end up with a predictor that is partially unfair with respect to both attributes, without any guarantee on either.

This is the existing behaviour of `WassersteinCorrector` when `protected_attrs` has more than one element. All attribute ECDFs are computed in `fit()` on the original predictions. The `transform()` loop applies corrections to the running output — which means the distributions don't match the calibration. For a single attribute it is correct. For $K \geq 2$ it is not.

---

## The Hu-Ratz-Charpentier fix

The correct algorithm, from Hu, Ratz & Charpentier (AAAI 2024), is a two-pass sequential procedure. For $K$ protected attributes:

**Fit phase:**

- Step 1: Fit the $A_1$ corrector on $f^*$. Compute $F_{a_1}^{(1)}$ from $f^*$, compute barycenter $\bar{Q}_1$. Apply to get $f_1 = T_{A_1}^{(1)}(f^*)$
- Step 2: Fit the $A_2$ corrector on $f_1$. Compute $F_{a_2}^{(2)}$ from $f_1$, compute barycenter $\bar{Q}_2$. Apply to get $f_2 = T_{A_2}^{(2)}(f_1)$
- Continue through $K$: each corrector's ECDF is calibrated on the output of the preceding step

**Transform phase:**

For a new prediction $f^*(x_i)$, apply the chain of fitted correctors in order: $f_1(x_i) \to f_2(x_i) \to \cdots \to f_K(x_i)$.

The key theorem (Hu et al. 2024, Theorem 1): the terminal predictor $f_K$ achieves Strong Demographic Parity with respect to all $K$ attributes simultaneously. Formally, $f_K \perp A_k$ for all $k = 1, \ldots, K$.

The order of attributes matters for intermediate predictions $f_1, \ldots, f_{K-1}$, but not for whether the terminal $f_K$ achieves joint SDP. You will get joint fairness regardless of whether you correct gender-first or ethnicity-first. The intermediate transport paths differ; the destination does not.

---

## What we have changed in insurance-fairness

The existing `WassersteinCorrector` calibrates all attribute ECDFs on $f^*$ in a single `fit()` call. This is correct for $K = 1$ and incorrect for $K \geq 2$.

We are adding `SequentialOTCorrector` to `insurance_fairness.optimal_transport`, which implements the Hu-Ratz-Charpentier two-pass algorithm. The only algorithmic difference from the naive sequential approach is that each corrector's ECDF is fitted on the previous step's output:

```python
from insurance_fairness.optimal_transport import SequentialOTCorrector
import polars as pl
import numpy as np

# D has columns: "gender_proxy", "ethnicity_proxy"
corrector = SequentialOTCorrector(
    protected_attrs=["gender_proxy", "ethnicity_proxy"],
    epsilon=0.0,           # full demographic parity
    n_quantiles=1000,
    log_space=True,        # work in log-premium space; recommended for pricing
    exposure_weighted=True,
)

# predictions: array of floats from your fitted pricing model
# D_calib: Polars DataFrame with the two protected attribute columns
corrector.fit(predictions=f_star, D_calib=D_calib, exposure=exposure)

# At inference time
f_fair = corrector.transform(predictions=f_star_test, D_test=D_test)
```

For a single protected attribute, `SequentialOTCorrector` produces identical output to `WassersteinCorrector`. The difference only materialises with $K \geq 2$.

The partial correction parameter works as before:

```python
corrector = SequentialOTCorrector(
    protected_attrs=["gender_proxy", "ethnicity_proxy"],
    epsilon=0.3,   # 70% of the way to demographic parity
)
```

The `WassersteinCorrector` remains in the API. It is the correct choice for single-attribute correction and is simpler to reason about. Use `SequentialOTCorrector` when you have two or more protected attributes and need the joint SDP guarantee.

---

## Checking that it works

Two diagnostics to run after fitting. First, per-group Wasserstein-1 distances before and after correction:

```python
# Before correction
from insurance_fairness.optimal_transport._utils import wasserstein_distance_1d

groups = D_calib["gender_proxy"].unique().to_list()
g0_mask = D_calib["gender_proxy"].to_numpy() == groups[0]
g1_mask = ~g0_mask
w1_before = wasserstein_distance_1d(
    np.log(f_star[g0_mask]), np.log(f_star[g1_mask]),
    exposure[g0_mask], exposure[g1_mask],
)

w1_after = wasserstein_distance_1d(
    np.log(f_fair[g0_mask]), np.log(f_fair[g1_mask]),
    exposure[g0_mask], exposure[g1_mask],
)

print(f"W1 gender_proxy: {w1_before:.4f} -> {w1_after:.4f}")
```

After full correction ($\varepsilon = 0$) the W1 distance should be close to zero — not exactly zero due to the discrete ECDF approximation with finite $n$, but below 0.001 on a book of 50,000+ policies is achievable. For partial corrections, W1 scales roughly linearly with $1 - \varepsilon$.

Second, verify that the $A_1$ fairness achieved in step 1 survives through step $K$. After the full sequential correction, check W1 for both attributes:

```python
for attr in ["gender_proxy", "ethnicity_proxy"]:
    col = D_test[attr].to_numpy()
    vals = np.unique(col)
    if len(vals) == 2:
        m0 = col == vals[0]
        m1 = ~m0
        w1 = wasserstein_distance_1d(
            np.log(f_fair[m0]), np.log(f_fair[m1]),
            exposure_test[m0], exposure_test[m1],
        )
        print(f"W1 {attr}: {w1:.4f}")
```

If $A_1$ W1 is materially above zero after the full correction, either the calibration set is too small (see below) or there is something wrong with the implementation.

---

## The exposure weighting question

The barycenter construction weights each group's quantile function by portfolio exposure share $\omega_a$. This has a non-obvious consequence: if your calibration data has 90% male and 10% female (by exposure), the barycenter will sit very close to the male distribution, and the correction moves female predictions substantially more than male ones.

Whether this is desirable depends on your objective. If you want the corrected distribution to match what the portfolio-weighted average risk would see, exposure-weighting is correct. If you want to achieve strict distributional equality between groups regardless of their portfolio share, equal weights ($\omega_a = 1/|\mathcal{A}|$) are what you want.

The default in `SequentialOTCorrector` (and `WassersteinCorrector`) is `exposure_weighted=True`. For UK regulatory purposes, where you are demonstrating that the pricing distribution is not systematically worse for a protected group relative to a similar risk in another group, the portfolio-weighted version is the appropriate diagnostic.

---

## Calibration data size requirements

The ECDF approximation degrades with small samples. At the group level. Not the total sample size.

For a two-category attribute (e.g. binary gender proxy), you need roughly 1,000 observations per group for the ECDF to be sufficiently smooth that the quantile interpolation is not dominated by granularity artefacts. Below 500 per group, the W1 distance estimates are noisy and the correction will introduce artificial bumps in the corrected distribution.

This is a real constraint for ethnicity in UK insurance. If your ethnicity proxy (via postcode MOSAIC/Acorn cluster) identifies five groups, and one group accounts for 3% of your portfolio on a 20,000-policy calibration set, that is 600 observations — borderline. For portfolio sizes below 50,000 policies, consider collapsing to three groups (majority, two largest minority groups) before applying the correction.

Frequency/severity decomposition complicates this further. If you run separate corrections on frequency predictions and severity predictions, the combined premium is not the same as running the correction on the combined premium prediction directly. The OT correction is a non-linear function of the input distribution; it does not commute with multiplication. Our recommendation: apply OT correction to the combined pure premium prediction, not to frequency and severity separately.

---

## What demographic parity is not


Strong Demographic Parity means $P(f(X) \leq t | A = a)$ is identical for all $a$ — the prediction distribution is the same across protected groups. OT correction achieves this. It does not achieve conditional fairness: two policyholders with identical risk profiles but different protected attribute values can still receive different prices after OT correction, if they were in different parts of their respective group distributions.

Conditional fairness — equal prices for equal risks regardless of protected attribute — is what Lindholm (2022) marginalisation achieves and is the correct standard under:

- **Equality Act 2010 s.19**: indirect discrimination requires showing that a policy produces disproportionate disadvantage. Equal prices for equal risks means the characteristic cannot be causing the disproportionate effect
- **FCA Consumer Duty**: the fair value outcome requires that price reflects risk. A demographic parity target that prices identical risks differently because of group membership is not "fair value" in the FCA's sense

The correct UK pipeline is therefore:

1. **Lindholm marginalisation first** (`LindholmCorrector`): achieves conditional fairness — equal prices for equal risks. This addresses the UK regulatory requirement
2. **Sequential OT second, as a diagnostic** (`SequentialOTCorrector`): after Lindholm correction, check whether residual demographic parity gaps remain. If they do, the OT correction quantifies the remaining distributional gap and can apply a partial correction ($\varepsilon > 0$) to close it

OT correction as the primary fairness mechanism is theoretically elegant but practically misaligned with the UK regulatory standard. Lindholm first, OT diagnostic second.

```python
from insurance_fairness.optimal_transport import (
    LindholmCorrector,
    SequentialOTCorrector,
)

# Step 1: Lindholm marginalisation (conditional fairness)
lc = LindholmCorrector(
    protected_attrs=["gender_proxy", "ethnicity_proxy"],
    bias_correction="proportional",
)
lc.fit(model_fn=model.predict, X_calib=X_cal, D_calib=D_cal, exposure=exposure)
f_lindholm = lc.transform(model_fn=model.predict, X=X_test, D=D_test)

# Step 2: Sequential OT diagnostic — how much residual distributional gap remains?
ot = SequentialOTCorrector(
    protected_attrs=["gender_proxy", "ethnicity_proxy"],
    epsilon=0.5,   # partial: halfway between Lindholm output and full demographic parity
    log_space=True,
    exposure_weighted=True,
)
ot.fit(predictions=f_lindholm_cal, D_calib=D_cal, exposure=exposure)
f_final = ot.transform(predictions=f_lindholm_test, D_test=D_test)

print(ot.wasserstein_distances_before_)   # residual gap after Lindholm
print(ot.wasserstein_distances_after_)    # residual gap after sequential OT
```

If Lindholm marginalisation is working correctly, the Wasserstein distances before the OT step should already be small. The OT correction then acts as a trimming tool rather than the primary correction. On most UK motor books we have tested, the W1 gap after Lindholm is below 0.05 on a log scale — small enough that full OT correction at $\varepsilon = 0$ is not warranted, and the partial correction at $\varepsilon = 0.5$ moves the residual to below measurement noise.

---

## Protected attribute data in UK insurance

The practical constraint that undermines most of this: you often do not hold the protected attribute data at all.

Gender: under the EU Gender Directive (2012), UK motor insurers cannot use gender as a direct rating factor. Most insurers stopped collecting it at point of sale. What you have are proxy signals — vehicle type, NCD profile, telematics behaviour — that correlate with gender. The Wasserstein distance you are correcting is the gap in the prediction distribution induced by those proxies, not by gender directly.

Ethnicity: UK general insurance firms hold ethnicity data for a small fraction of their portfolios. What you have is postcode-based geodemographic clusters (MOSAIC, Acorn, CACI) that are correlated with ethnicity at the group level. The proxy relationship is noisy: a postcode cluster is not an individual ethnicity. Correcting for the cluster is a reasonable proxy for correcting for ethnicity but does not guarantee equality at the individual level.

Disability: not typically held or inferred. The OT framework does not extend gracefully to attributes you cannot observe.

The practical implication: when you implement sequential OT correction, your $A_k$ are likely proxy variables, not the protected characteristics themselves. The fairness guarantee (joint SDP w.r.t. $A_1, \ldots, A_K$) holds for the proxies you specify. Whether achieving SDP on the proxies achieves SDP on the underlying characteristics depends on the proxy relationship — which you should test and document.

---

## The order question in practice

The AAAI 2024 theorem says attribute order does not affect whether you achieve joint SDP. In practice, the intermediate predictions will differ, and if you are running the OT correction as a partial correction ($\varepsilon > 0$), order can affect the final result because the blending step is applied after the full $K$-step chain.

Our recommendation for UK insurance: order by regulatory priority, most salient first. If you have a well-founded view that the gender proxy gap is larger than the ethnicity proxy gap (measured in W1 distance before any correction), correct gender first. The first correction makes the largest move; subsequent corrections make smaller adjustments. This aligns the partial correction result more closely with the attributes where the regulatory exposure is highest.

---

## Installation and source

```bash
uv add insurance-fairness

# or with all optional dependencies
uv add "insurance-fairness[all]"
```

`SequentialOTCorrector` ships in `v0.4.0` alongside the existing `WassersteinCorrector`. The single-attribute API is unchanged. Source: [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness).

The Hu-Ratz-Charpentier paper is at arXiv:2309.06627. The original EquiPy Python package implementing the same algorithm is at arXiv:2503.09866 (Fernandes Machado, Charpentier et al., March 2025). Our implementation extends the algorithm with exposure weighting, log-space corrections appropriate for multiplicative pricing models, and integration with the Lindholm marginalisation pipeline already present in `insurance-fairness`.

---

## Related posts

- [Discrimination-Free Insurance Pricing: The Lindholm Approach](https://burning-cost.github.io/fairness/2026/03/28/discrimination-free-insurance-pricing-lindholm/) — the Lindholm marginalisation that should precede OT correction in any UK pipeline
- [Proxy Discrimination in Motor Pricing: What FCA PS25/21 Requires](https://burning-cost.github.io/regulation/2026/03/28/fca-ps2521-proxy-discrimination-motor-pricing/) — the regulatory framing for indirect discrimination
