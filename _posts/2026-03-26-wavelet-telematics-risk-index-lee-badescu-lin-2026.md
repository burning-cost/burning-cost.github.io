---
layout: post
title: "Portfolio-Anchored Telematics Risk Scoring with Wavelets"
date: 2026-03-26
categories: [telematics, pricing]
tags: [telematics, wavelets, modwt, bayesian, poisson-gamma, ubi, motor, pricing, python, fca-consumer-duty, credibility]
description: "Lee, Badescu, and Lin (2026) replace ad-hoc event counts with a principled actuarial risk index: MODWT decomposes the acceleration signal, a Gaussian-Uniform mixture anchors tail rarity to the portfolio, and Poisson-Gamma conjugate updating gives a closed-form score that improves with every trip."
---

Current UBI pricing is, mostly, artisanal. Telematics providers package some combination of harsh braking counts, cornering events, speeding fractions, and night driving into a proprietary composite score. The weights are chosen by domain experts, validated against loss ratios on historical data, and then frozen. The problem is not that these scores are inaccurate — they often work reasonably well. The problem is that they are not actuarial. There is no principled connection between the score and claim probability, no natural way to set how much a single harsh manoeuvre should count versus sustained moderate aggression, and no framework for handling the cold-start problem beyond arbitrary dampening constants.

Lee, Badescu, and Lin (arXiv:2603.15839, submitted to ASTIN Bulletin, March 2026) propose something better: a risk index derived end-to-end from the raw acceleration signal, anchored to the portfolio distribution, and updated sequentially via conjugate Bayes as trip history accumulates. The machinery is unfamiliar to most pricing teams — maximal overlap discrete wavelet transform, Gaussian-Uniform mixtures, multi-layer tail counts — but the output is a single trip-level score that plugs directly into a multiplicative GLM. The maths is tractable and the implementation is feasible in Python with standard packages plus one custom piece.

---

## The problem with event counting

Every telematics scoring system starts by defining a harsh event: an acceleration reading that exceeds some threshold in g-force, held for more than some minimum duration. The threshold is chosen to separate genuine harsh manoeuvres from road surface noise. The duration filter reduces false positives from potholes.

There are two things wrong with this approach.

First, the threshold is arbitrary relative to the portfolio. A 0.3g threshold on a fleet of elderly drivers in Norfolk will classify as harsh many events that a young male fleet in Glasgow barely notices, because the portfolio distributions are different. The threshold is set in physics space, not probability space.

Second, event counting treats a count of five harsh events the same whether those five events are marginal manoeuvres just above the threshold or extreme manoeuvres deep in the tail. There is no severity dimension. All events are equal once they cross the line.

Lee-Badescu-Lin address both problems simultaneously. The severity of a manoeuvre is measured by its rarity within the portfolio distribution, not by its absolute g-force. And rarity is computed relative to a mixture model fitted to the full portfolio signal — so the same g-force gets different severity scores on different books.

---

## Step 1: MODWT on the acceleration signal

The raw input is per-second longitudinal acceleration (m/s²) from OBD or smartphone. For trip _i_ of length _T_ᵢ_ seconds, call this _X_{i,t}_.

The paper applies the maximal overlap discrete wavelet transform (MODWT) with a Daubechies D4 filter at _J_ = 6 decomposition levels. MODWT is the non-decimating version of the standard DWT: unlike DWT, it produces output the same length as the input at every level. This matters for variable-length trips — you get a coefficient for every second of the trip, with no alignment artefacts at boundaries.

At decomposition level _j_, the wavelet coefficients are:

```
W̃_{i,j,t} = Σ_{l=0}^{L_j−1} h̃_{j,l} · X_{i,(t−l) mod T_i}
```

where _h̃_{j,l}_ are the rescaled Daubechies D4 filter coefficients. The six levels cover timescales from roughly 1–2 seconds (sharp braking) up to ~64 seconds (sustained aggressive driving). A harsh brake shows up at fine scales; a sustained aggressive episode at coarse scales.

Rather than keeping all six levels separately, the paper collapses them with a maximum rule:

```
C_{i,t} = max_{j ∈ {1,...,J}} |W̃_{i,j,t}|
```

This gives a single scalar _C_{i,t}_ for each second of each trip — the most extreme driving event at any temporal scale at that moment. Prior wavelet work on telematics (Wüthrich 2017, same D4 filter) typically used wavelet energy per level, which averages away the temporal localisation. The maximum rule preserves it.

In Python, using PyWavelets:

```python
import pywt
import numpy as np

def modwt_max_coeff(accel: np.ndarray, wavelet: str = "db2", levels: int = 6) -> np.ndarray:
    """
    MODWT (via stationary WT) of a 1D acceleration signal.
    Returns per-second max absolute wavelet coefficient across all levels.

    accel : 1D float array, per-second acceleration in m/s²
    wavelet : 'db2' is Daubechies D4 (4-tap) in pywt convention.
              WARNING: pywt 'db4' is an 8-tap filter, NOT D4. Use 'db2' for the paper's D4.
    levels : decomposition depth J=6 recommended
    """
    # swt requires length to be divisible by 2^levels
    n = len(accel)
    pad = int(2**levels * np.ceil(n / 2**levels)) - n
    padded = np.pad(accel, (0, pad), mode="reflect")

    coeffs = pywt.swt(padded, wavelet=wavelet, level=levels, norm=True)
    # coeffs is a list of (cA, cD) tuples, one per level
    # cD contains the wavelet (detail) coefficients
    detail_stack = np.stack([cD[:n] for _, cD in coeffs], axis=0)
    return np.max(np.abs(detail_stack), axis=0)  # shape (n,)
```

The `norm=True` argument applies the MODWT rescaling (divides by √2 at each level), which is what makes the transform energy-preserving and the coefficients comparable across levels.

---

## Step 2: Portfolio-anchored severity via Gaussian-Uniform mixture

You now have a per-second score _C_{i,t}_ for every trip in the portfolio. Pool them all into one dataset and fit a mixture model.

The model is not a standard Gaussian mixture. The centre of the distribution (normal driving) is captured by _G_ = 2 Gaussian components. The tails — harsh braking on the left, harsh acceleration on the right — are captured by ordered, non-overlapping Uniform layers:

```
f(c; η) = Σ_{m'=1}^{M⁻} π_{m'}⁻ · U(c; θ_{m'}⁻)
         + Σ_{g=1}^{G}   π_g · φ(c; θ_g)
         + Σ_{m=1}^{M⁺}  π_m⁺ · U(c; θ_m⁺)
```

where _U(c; θ_m)_ is Uniform on the interval _[u_m, u_{m+1}]_ and the intervals are ordered — layer _m_ is always deeper in the tail than layer _m_−1. BIC-selected on the UAH-DriveSet validation data: _G_ = 2, _M⁻_ = 4 left layers (braking), _M⁺_ = 5 right layers (acceleration).

The key property is that each mixing weight _π_m_ is the fraction of all portfolio seconds that fall in that severity band. The deepest right-tail layer in the fitted model has _π_ = 0.092% — about 1 in 1,000 seconds of portfolio driving reaches that level of harsh acceleration. The shallowest left-tail layer (mild harsh braking) has _π_ = 1.591%.

These are actuarially meaningful numbers. They are the portfolio's natural severity scale, expressed in probability space.

Fitting this model — the MU-MEMR algorithm in the paper — requires a custom EM with isotonic regression constraints on the Uniform layer boundaries. There is no sklearn class that does this. The implementation is the heaviest piece of the framework, roughly 300–400 lines of EM code using `scipy.optimize.isotonic_regression`. But it fits once on the full portfolio and then freezes; it does not need re-fitting as new trips arrive.

---

## Step 3: Multi-layer tail counts and Poisson-Gamma updating

Once the mixture model is fitted, you can map every second of every trip to a layer. For trip _i_, the count of seconds in severity layer _m_ is:

```
N_{im} = Σ_{t=0}^{E_i−1} 1{C_{i,t} ∈ Θ_m}
```

where _Θ_m_ is the interval for layer _m_ and _E_i_ is trip length in seconds. This is the **multi-layer tail count** (MLTC): not a single event count, but a vector of counts across the severity spectrum.

The MLTC sits inside a Poisson-Gamma conjugate model. The per-second rate of tail events in layer _m_ for trip _i_ is _λ_{im}_, with prior fitted from the portfolio via Winsorised moment matching:

```
N_{im} | λ_{im}, E_i ~ Poisson(E_i · λ_{im})
λ_{im}              ~ Gamma(α_{0m}, β_{0m})
```

The trip-level posterior is:

```
λ_{im} | E_i, N_{im} ~ Gamma(α_{0m} + N_{im},  β_{0m} + E_i)
```

After _k_ trips, the driver-level sequential update is:

```
α_{am}^(k) = α_{am}^(k−1) + N_{km}
β_{am}^(k) = β_{am}^(k−1) + E_k
```

The posterior mean — the credibility-weighted rate estimate — is:

```
λ̂_{am}^(k) = [β_{am}^(k−1) / (β_{am}^(k−1) + E_k)] · λ̂_{am}^(k−1)
            + [E_k           / (β_{am}^(k−1) + E_k)] · (N_{km}/E_k)
```

This is a credibility formula. The weight on the new trip's empirical rate _Z_k = E_k / (β_{am}^(k−1) + E_k)_ increases as total accumulated exposure grows. A driver with 200 trips has much higher _Z_ than a driver with 2.

The severity layers are combined into the final risk index by inverse-probability weighting:

```
w_m = (1/π_m)^γ / Σ_l (1/π_l)^γ
```

with _γ_ = 1.7 in the paper's experiments. Rarer layers get exponentially more weight — a second in the deepest tail layer contributes roughly 20× more to the index than a second in the shallowest layer.

The closed-form driver-level risk index after _k_ trips is:

```
Ŝ_a^(k) = Σ_m w_m · α_{am}^(k) / β_{am}^(k)
```

For a new driver with no trip history, _α_{am}^(0) = α_{0m}_ and _β_{am}^(0) = β_{0m}_, so _Ŝ_a^(0)_ equals the portfolio prior mean. There is no cold-start problem — new drivers start at average risk and update from there.

---

## Putting the pipeline together

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class DriverState:
    alpha: np.ndarray  # shape (n_layers,) — posterior alpha per layer
    beta:  np.ndarray  # shape (n_layers,) — posterior beta per layer


class WaveletRiskIndex:
    def __init__(self, mixture_model, prior_alpha, prior_beta, severity_weights):
        """
        mixture_model   : fitted MU-MEMR model, with .layer_intervals list
        prior_alpha     : portfolio Gamma prior alpha, shape (n_layers,)
        prior_beta      : portfolio Gamma prior beta,  shape (n_layers,)
        severity_weights: inverse-probability weights, shape (n_layers,)
        """
        self.mixture   = mixture_model
        self.alpha0    = prior_alpha
        self.beta0     = prior_beta
        self.weights   = severity_weights

    def new_driver(self) -> DriverState:
        return DriverState(alpha=self.alpha0.copy(), beta=self.beta0.copy())

    def score_trip(self, accel: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (tail_counts_per_layer, trip_risk_index)."""
        C = modwt_max_coeff(accel)
        exposure = len(accel)
        counts = np.array([
            np.sum((C >= lo) & (C < hi))
            for lo, hi in self.mixture.layer_intervals
        ], dtype=float)
        alpha_post = self.alpha0 + counts
        beta_post  = self.beta0  + exposure
        score = float(self.weights @ (alpha_post / beta_post))
        return counts, score

    def update_driver(self, state: DriverState, accel: np.ndarray) -> DriverState:
        """Sequential Bayesian update after one trip."""
        counts, _ = self.score_trip(accel)
        exposure   = len(accel)
        return DriverState(
            alpha = state.alpha + counts,
            beta  = state.beta  + exposure,
        )

    def driver_score(self, state: DriverState) -> float:
        return float(self.weights @ (state.alpha / state.beta))
```

The driver-level update is nine floating-point additions and one dot product. Real-time feasible at any scale.

---

## Honest comparison with the HMM approach

We covered hidden Markov model telematics scoring in [`insurance-telematics`](/insurance-telematics/) — the `DrivingStateHMM` uses trip-aggregate features (harsh braking rate, mean speed, night fraction) to infer a latent regime sequence and expresses driver risk as state-fraction covariates in a Poisson GLM. On a three-state DGP it beats raw averages by 5–10 Gini points.

The wavelet approach and the HMM approach operate at different levels of the data hierarchy. This is the clearest way to compare them:

| Dimension | HMM (DrivingStateHMM) | Wavelet (Lee-Badescu-Lin) |
|---|---|---|
| Input granularity | Trip-aggregate features | Per-second raw acceleration |
| Severity representation | None — state fractions are frequency, not magnitude | Multi-layer tail counts — explicit severity spectrum |
| Cold-start handling | Requires minimum trips; arbitrary dampening | Conjugate prior gives portfolio mean immediately |
| Interpretability | Latent states have natural labels (cautious/normal/aggressive) | Severity layers are probability quantiles of portfolio distribution |
| Claim linkage | Indirect — state fraction → GLM → claim probability | Direct — tail rate → Poisson rate → claim probability |
| Implementation complexity | Moderate — standard HMM packages | High — custom MU-MEMR fitting required |

We think the wavelet framework is the stronger approach for a book with raw per-second data available. The explicit severity weighting solves a real problem that HMMs cannot: distinguishing a driver who accumulates their harsh-event exposure via dozens of mild events from one who generates the same count via a handful of genuinely extreme manoeuvres. The Bayesian credibility structure is also cleaner — the cold-start behaviour is principled rather than engineered.

The HMM is still useful when you only have trip-aggregate data (many UK black box contracts batch-report rather than streaming raw signal), or when interpretable latent states are a regulatory requirement. And the two are not mutually exclusive: the WRI output _Ŝ_a^(k)_ could feed directly into a GLM alongside HMM state fractions, giving both the severity-spectrum signal and the temporal-regime signal.

---

## The FCA Consumer Duty angle

UK pricing teams face an increasing burden under FCA Consumer Duty (PS22/9) and the legacy of the GIPP pricing reforms: demographic proxies — even indirect ones — are under scrutiny. Postcode remains legal for motor; gender is banned as a direct rating factor (since the 2012 ECJ ruling); age-related correlates face scrutiny under proxy discrimination rules; anything that functions as a proxy for protected characteristics needs an explicit actuarial justification.

The Lee-Badescu-Lin framework is purely behavioural. The risk index is derived from the acceleration time series on each trip, anchored to the portfolio distribution of acceleration events, and updated only from the driver's own trip history. No demographics enter the model at any stage. The prior _Gamma(α_{0m}, β_{0m})_ is estimated from portfolio-level tail rates — it reflects the distribution of driving behaviour, not of driver characteristics.

This is a genuinely useful property. A risk index that is provably behaviour-only is defensible under Consumer Duty in a way that a composite score derived partly from pricing segmentation data is not. If the regulator asks "what does this score reflect about the individual's behaviour?" the answer is exact and auditable — it is the posterior mean rate of per-second tail events in each severity layer, weighted by portfolio rarity.

That will not always be enough: if the telematics score correlates with protected characteristics at the portfolio level (e.g. because night driving correlates with occupation which correlates with age), the score is still potentially discriminatory regardless of its derivation. But starting from a behaviour-only foundation at least removes one source of concern.

---

## What it takes to implement

The honest assessment: most of this is straightforward, one piece is not.

**MODWT feature extraction** — two or three function calls with PyWavelets. Add `pywt` as a dependency, write the `modwt_max_coeff` function above, done. An afternoon.

**Poisson-Gamma updating** — pure numpy, seven lines of arithmetic. A morning to write and test thoroughly.

**MU-MEMR mixture fitting** — this is the hard part. No package provides ordered Uniform mixtures with isotonic constraints. Expect to write roughly 300–400 lines of EM code, with `scipy.optimize.isotonic_regression` for the monotonicity enforcement and a BIC grid search over left-tail and right-tail layer counts. Realistic estimate: 2–3 weeks of engineering time to get a robust, tested implementation. The good news is it runs once on the portfolio and then freezes.

**Data requirements** — raw per-second (1Hz) longitudinal acceleration. For MODWT at _J_ = 6 levels, minimum trip length is 64 seconds. Most modern OBD and smartphone telematics collect at 1Hz or higher; the UAH-DriveSet used in the paper runs at ~10Hz and is aggregated down.

The computational profile is sensible: MODWT on a 10-minute trip at 1Hz takes 600 × 6 multiply-adds (negligible), MU-MEMR fits once on the full portfolio (minutes on a laptop), and per-driver updates are constant-time per trip.

---

The paper is at [arXiv:2603.15839](https://arxiv.org/abs/2603.15839). We plan to implement the wavelet risk index module for `insurance-telematics` v0.2.0 — the API sketch in KB has the design.

- [HMM-Based Telematics Risk Scoring for Insurance Pricing](/2026/03/13/insurance-telematics/) — the current `insurance-telematics` library that the wavelet risk index will extend, with CTHMM-based driving state features
- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/) — the conjugate Bayes updating in the Lee et al. framework shares its structure with Bühlmann-Straub credibility
