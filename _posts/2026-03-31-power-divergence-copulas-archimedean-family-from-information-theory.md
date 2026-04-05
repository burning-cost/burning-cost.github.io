---
layout: post
title: "A New Archimedean Family from Information Theory: Power-Divergence Copulas"
date: 2026-03-31
categories: [techniques]
tags: [copulas, archimedean, power-divergence, phi-divergence, tail-dependence, joint-severity, commercial-property, lloyds, danish-fire, information-theory, Cressie-Read, KL-divergence, zero-set, scipy, python, Pearse-Bondell-2025, arXiv-2510-06177]
description: "Pearse & Bondell (arXiv:2510.06177, October 2025) derive a new Archimedean copula family from Cressie-Read phi-divergences. The generator table maps KL divergence to one copula, reverse KL to another, and the full power-divergence family interpolates between them. One structural property makes it unlike anything in the standard toolkit: upper tail dependence is fixed at 0.586 regardless of the overall dependence level. The Danish fire dataset, which nine standard copulas fail to fit, passes goodness-of-fit only with the PD copula."
author: burning-cost
---

Standard references give you four Archimedean families to choose from: Clayton, Frank, Gumbel, Joe. For most insurance applications this is enough. Clayton handles lower tail dependence (joint large losses); Gumbel handles upper tail dependence; Frank is symmetric with no tail structure; Joe sits somewhere between Frank and Gumbel. The families are not interchangeable but they are, collectively, a complete toolkit for most frequency-severity and aggregate modelling problems.

Pearse and Bondell (arXiv:2510.06177, University of North Carolina, October 2025) argue there is a gap. Their new family — power-divergence copulas — comes from a direction nobody had tried: using the convex functions defining Csiszar phi-divergences directly as Archimedean generators. The result is a family with a fixed upper tail dependence coefficient regardless of the parameter value, a nontrivial zero set, and a clean information-theoretic interpretation. The Danish fire dataset, which every one of Clayton, Frank, Gumbel, Joe, Galambos, Husler-Reiss, Tawn, and the Gaussian copula fail on goodness-of-fit, passes with the PD copula.

That result is specific enough to be worth understanding in detail.

---

## The generator idea

Every Archimedean copula is built from a generator function phi: [0,1] -> [0, inf) that is strictly decreasing, strictly convex, and satisfies phi(1) = 0. Given a generator, the bivariate copula is:

```
C(u1, u2) = phi^{-1}(phi(u1) + phi(u2))
```

where phi^{-1} is the pseudo-inverse — returning 0 whenever the argument exceeds phi(0). Clayton uses phi(x) = (x^{-theta} - 1)/theta. Frank uses phi(x) = -log((e^{-theta*x} - 1)/(e^{-theta} - 1)). Gumbel uses phi(x) = (-log x)^theta.

None of these come from information theory. The Pearse-Bondell insight is that the Cressie-Read power-divergence functions from multinomial goodness-of-fit testing (Cressie & Read, JRSS-B, 1984) happen to satisfy all the Archimedean generator requirements. The family is parameterised by lambda in (-inf, +inf):

```
phi_lambda(x) = (x^(lambda+1) - x + lambda*(1-x)) / (lambda*(lambda+1))   for lambda != -1, 0
              = 1 - x + x*log(x)                                            for lambda = 0  [KL divergence]
              = x - 1 - log(x)                                              for lambda = -1 [reverse KL]
```

Each special value of lambda corresponds to a named divergence. KL divergence (lambda = 0) and reverse KL (lambda = -1) give particular copulas. Lambda = 1 is the chi-squared divergence. Lambda = -0.5 is the Hellinger distance. The full family interpolates continuously between them.

The generator satisfies phi_lambda(1) = 0, phi_lambda'(1) = 0, and phi_lambda''(1) > 0. The normalisation phi_lambda'(1) = 0 is a regularity condition that simplifies the family: it ensures the generator has a consistent scale across all lambda values, which is what makes the parameter range comparable. It is not a necessary condition for staying strictly inside the Frechet-Hoeffding bounds — Clayton's generator has phi'(1) ≠ 0 and Clayton copulas also lie strictly inside those bounds. What matters for the Frechet-Hoeffding property is that the generator is strictly convex and equals zero at x = 1, both of which hold for phi_lambda across the entire lambda range.

---

## Kendall's tau and estimation

Kendall's tau is strictly decreasing in lambda: tau(-inf) = 1, tau(+inf) = -1. The two tractable special cases are:

- lambda = 0: tau = 3 - 4*log(2) ≈ 0.227
- lambda = -1: tau = 7 - 2*pi^2/3 ≈ 0.420

For any observed sample tau, method-of-moments estimation reduces to a one-dimensional root-find over lambda. This is clean, consistent, and asymptotically unbiased — simpler estimation than most vine copula families.

---

## The zero set: not all losses can simultaneously be small

The most structurally unusual property of PD copulas is the zero set — the region in [0,1]^2 where the copula is identically zero.

For Clayton, Frank, Gumbel, and Joe, the zero set is just the boundary of the unit square: {(u,0)} and {(0,v)}. That means joint outcomes below any interior threshold have positive probability. For PD copulas with lambda > -1, this is not true. There is a nontrivial zero curve inside [0,1]^2, defined by:

```
{(u1, u2) : phi_lambda(u1) + phi_lambda(u2) = phi_lambda(0)}
```

Points below this curve have C_lambda = 0. The copula places zero probability mass on joint outcomes where both margins are simultaneously small.

For lambda > 0, the zero set is substantial — as lambda increases, it expands to fill the entire lower-left triangle. Theorem 3.2 of Pearse & Bondell shows that for lambda > 0, the copula has a singular component supported on the zero curve, with C-measure lambda/(lambda+1). At lambda = 1, half the copula's mass is concentrated on this curve. At lambda = 0.5, a third is.

The insurance interpretation is direct. Consider two loss types that share a common triggering event — a large fire causing both material damage and business interruption. A fire large enough to trigger a claim is large enough to affect both. There is a structural floor: small material losses and small business interruption losses cannot co-occur given a genuine loss event. The zero set captures this. Standard Archimedean families cannot.

For lambda <= -1, the copula is absolutely continuous with a density everywhere and the zero set reduces to the boundary. This is the multivariate-safe region: it is also the most tractable range for implementation.

---

## The fixed upper tail: 0.586 regardless of lambda

The tail dependence coefficients are:

- **Lower tail:** T_L = 2^{1/(lambda+1)} for lambda < -1; T_L = 0 for lambda >= -1.
- **Upper tail:** T_U = 2 - sqrt(2) ≈ **0.586 for all lambda in (-inf, +inf)**.

The upper tail coefficient is constant across the entire parameter range. This is unique among standard Archimedean families. Gumbel has T_U = 2 - 2^{1/theta}, which varies from 0 to 1 as theta increases. Clayton has T_U = 0 for all theta. Frank has T_U = 0. Joe's T_U varies.

The implication for modelling: a PD copula always exhibits moderate upper tail dependence. You cannot tune it away. If you need low upper tail dependence — independent large losses — the PD copula is the wrong choice. If you accept that co-extreme large losses are a structural feature of the peril or line you are modelling, then fixing T_U = 0.586 a priori removes one parameter from the estimation problem. You know in advance that joint large losses occur together at a moderate rate, and lambda governs everything else.

For aggregate loss modelling across correlated perils — cat event scenarios, storm affecting multiple commercial properties in a Lloyd's syndicate — this is a defensible fixed parameter. You are saying the upper tail behaves like T_U = 0.586 because that is what Archimedean structure implies, not because you estimated it from thin data.

---

## The Danish fire result

The classical Danish fire dataset (Embrechts, Kluppelberg, Mikosch, 1997) records bivariate losses for large fires in Denmark: material damage and business interruption or profit losses per event. It is a standard benchmark for bivariate copula fitting in extreme value statistics.

The observed Kendall's tau is 0.361, which inverts to lambda_hat = -0.647. Pearse & Bondell fit nine copulas to this dataset and run parametric bootstrap goodness-of-fit at 5% significance:

| Copula | p-value | Pass |
|--------|---------|------|
| PD (lambda = -0.647) | 0.078 | YES |
| Clayton | < 0.05 | NO |
| Frank | < 0.05 | NO |
| Joe | < 0.05 | NO |
| Gumbel | < 0.05 | NO |
| Galambos | < 0.05 | NO |
| Husler-Reiss | < 0.05 | NO |
| Tawn | < 0.05 | NO |
| Gaussian | < 0.05 | NO |

The PD copula is the only one that passes. The fitted lambda = -0.647 is in the absolutely continuous region (-1 < lambda <= 0), so the density exists everywhere and the singular component does not apply — but the zero set structure is present and the copula is not radially symmetric. The combination of moderate upper tail dependence (T_U = 0.586) and an asymmetric interior zero curve is what distinguishes it from the nine alternatives.

The insurance logic is clean. A fire large enough to disrupt business also causes material damage — both losses are bounded away from zero given the event. And when fires are large, losses in both categories tend to be correlated at exactly the moderate level that T_U = 0.586 describes.

---

## UK application

The honest answer is: this is a commercial property and Lloyd's tool, not a personal lines motor tool.

For UK personal lines motor, the joint modelling problem is frequency-severity dependence — a discrete frequency variable and a continuous severity variable. That calls for a discrete-continuous copula. Our `insurance-frequency-severity` library uses the Sarmanov copula for this, which handles negative dependence (motor tau roughly -0.05 to -0.20 in UK data) well. PD copulas are defined for continuous-continuous joints only. They do not belong in the motor frequency-severity stack.

The natural UK application is multi-peril joint severity in home insurance or commercial property:

**Home fire + escape of water.** A house fire triggers suppression water damage. The joint (fire severity, escape of water severity) has a plausible zero set: a genuine fire event forces both losses above a threshold. Small fire severity and small water severity cannot co-occur in this conditioned dataset.

**Subsidence + building fabric.** Material ground movement elevates both foundation and fabric costs. Near-zero for both simultaneously is structurally unlikely given a confirmed subsidence event.

**Flood + contents.** Serious flooding produces correlated floor-level and contents damage with a structural joint floor.

**Lloyd's large peril severity.** A storm affecting multiple commercial properties in a syndicate produces upper tail co-movements that T_U = 0.586 represents reasonably. Given that the exact upper tail parameter is always poorly estimated from thin data, having it fixed by mathematical structure is an advantage rather than a limitation.

What is required to use this in practice: large peril-level claim files with both occurrence indicators and per-peril severity, and a sufficient sample of co-occurrence events. For personal lines this is demanding — you likely need 500+ co-events for stable estimation. Commercial property and Lloyd's data typically provide this.

For personal lines motor frequency-severity: use Sarmanov. For home joint severity or commercial property: PD copulas are the right tool.

---

## Python implementation

No Python package implements power-divergence copulas as of March 2026. pyvinecopulib does not include them. The `copulas`, `copulalib`, and `statsmodels` packages do not include them. The implementation requires only scipy and numpy.

The core is around 25 lines:

```python
import numpy as np
from scipy.optimize import brentq

def phi(x, lam):
    """Power-divergence generator."""
    if lam == 0:
        return 1.0 - x + x * np.log(x)
    elif lam == -1:
        return x - 1.0 - np.log(x)
    else:
        return (x**(lam + 1) - x + lam * (1 - x)) / (lam * (lam + 1))

def phi_inv(t, lam):
    """Pseudo-inverse via root-finding. Returns 0 beyond the zero curve."""
    if t >= phi(1e-12, lam):   # t >= phi(0+): point is in zero set
        return 0.0
    if lam == 0:
        # phi(x) = 1 - x + x*log(x); solve 1 - x + x*log(x) = t
        # The general formula degenerates at lam=0, so root-find directly.
        f = lambda x: 1.0 - x + x * np.log(x) - t
        return brentq(f, 1e-12, 1.0)
    elif lam == -1:
        # phi(x) = x - 1 - log(x); solve x - 1 - log(x) = t
        # Equivalent to Lambert W: x - log(x) = 1 + t, but brentq is simpler.
        f = lambda x: x - 1.0 - np.log(x) - t
        return brentq(f, 1e-12, 1.0)
    else:
        f = lambda x: x**(lam + 1) - (lam + 1) * x + lam - lam * (lam + 1) * t
        return brentq(f, 1e-12, 1.0)

def pd_copula(u1, u2, lam):
    """Bivariate power-divergence copula C_lambda(u1, u2)."""
    return phi_inv(phi(u1, lam) + phi(u2, lam), lam)
```

Closed forms for phi_inv exist at lambda in {-2, -1, -0.5, 1, 2, 3} — for lambda = -1, the Lambert W function via `scipy.special.lambertw`; for lambda = -2, the quadratic formula. For all other lambda, `brentq` requires roughly 50 function evaluations per call, making vectorised evaluation over a 10,000-pair dataset about 100 times slower than Clayton or Frank. For a full goodness-of-fit bootstrap over 100,000 policies this is slow. For fitting on a dataset of thousands of co-event pairs, it is workable.

Method-of-moments estimation:

```python
from scipy.stats import kendalltau
from scipy.optimize import brentq

def tau_from_lambda(lam, n_quad=500):
    """Kendall's tau via numerical integration of phi(x)/phi'(x)."""
    # tau = 1 + 4 * integral_0^1 phi(x)/phi'(x) dx
    # Use Gauss-Legendre quadrature
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(n_quad)
    x = 0.5 * (nodes + 1)            # map [-1,1] to [0,1]
    w = 0.5 * weights
    dx = 1e-7
    phi_prime = (np.array([phi(xi + dx, lam) for xi in x]) -
                 np.array([phi(xi - dx, lam) for xi in x])) / (2 * dx)
    phi_vals = np.array([phi(xi, lam) for xi in x])
    integrand = phi_vals / phi_prime
    return 1.0 + 4.0 * np.dot(w, integrand)

def fit_pd_copula(data):
    """Fit lambda by method of moments from bivariate data."""
    tau_obs, _ = kendalltau(data[:, 0], data[:, 1])
    lam_hat = brentq(lambda lam: tau_from_lambda(lam) - tau_obs, -10.0, 10.0)
    return lam_hat
```

The tau formula is a standard Archimedean result: tau = 1 + 4 * integral_0^1 phi(x)/phi'(x) dx. The integral has no closed form for general lambda but converges quickly with 500-point Gauss-Legendre quadrature.

---

## Multivariate restrictions

One practical constraint: the bivariate PD copula cannot be extended to all dimensions freely. The Archimedean multivariate extension to d variables requires the generator to be d-monotone. The restrictions are:

| lambda range | Max safe dimension | Notes |
|---|---|---|
| lambda > 0 | 2 | Singular component; not 3-monotone |
| -0.5 < lambda <= 0 | 2 | Not 3-monotone |
| -1 < lambda <= -0.5 | 3 | 3-monotone but not 4-monotone |
| lambda = -1 | All d | Completely monotone; closed form via Lambert W |
| lambda = -2 | All d | Completely monotone |
| lambda < -1 (general) | >= 3 | Conjectured completely monotone; not yet proved |

For bivariate applications — which cover all the UK home and commercial property use cases above — the full lambda range is available. For multivariate vine copulas using PD pair-copulas, restrict to lambda <= -1 and rely on the proved cases of -1 and -2 until the general conjecture is settled.

---

## Why we are not building this

The mathematical story is excellent. The Danish fire result is clean and compelling. The implementation is pure scipy — no new dependencies, no C extensions, no version risk. So why no new library?

The honest score: UK personal lines relevance is weak. Our primary audience is UK personal lines pricing actuaries working on motor and home. Motor frequency-severity is well-served by Sarmanov. Home joint peril severity is the natural application but it requires large co-event datasets that are unusual in personal lines. The commercial property and Lloyd's use case is real but represents a smaller segment of our readership, and the data requirements are demanding.

The integration question is also awkward. PD copulas do not belong in `insurance-frequency-severity` (discrete-continuous, Sarmanov handles it). They do not easily slot into `insurance-copula` either — pyvinecopulib does not support custom pair-copula families in its C++ API, so adding PD copulas as vine pair-copulas would require a parallel vine implementation. The cleanest option would be a standalone repo, which fragments the portfolio further for a niche use case.

Three things would flip this to a build decision:

1. pyvinecopulib adds a Python-defined custom BiCop interface — would allow clean PD pair-copula integration into `insurance-copula`
2. A UK commercial property or Lloyd's syndicate produces a case study showing PD copulas outperform existing families on UK data
3. The general complete monotonicity conjecture (lambda < -1 beyond -2) gets proved, enabling multivariate use across the full negative lambda range

Until then: blog, not build.

---

## What to read

- Pearse, A.R. & Bondell, H. (2025). "Power-Divergence Copulas." arXiv:2510.06177.
- Cressie, N. & Read, T.R.C. (1984). "Multinomial goodness-of-fit tests." *JRSS-B* 46(3), 440–464. The original source for the phi_lambda family.
- Csiszar, I. (1967). "Information-type measures of difference of probability distributions." *Studia Sci. Math. Hungar.* 2, 299–318. The general phi-divergence framework.
- Embrechts, P., Kluppelberg, C. & Mikosch, T. (1997). *Modelling Extremal Events*. The Danish fire dataset is reproduced and analysed here.
- Genest, C. & Rivest, L.-P. (1993). "Statistical inference procedures for bivariate Archimedean copulas." *JASA* 88(423), 1034–1043. The Kendall's tau inversion approach for Archimedean estimation used throughout the paper.

The full paper is at [arXiv:2510.06177](https://arxiv.org/abs/2510.06177).
