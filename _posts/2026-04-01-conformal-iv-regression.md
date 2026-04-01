---
layout: post
title: "Prediction Intervals That Survive an Instrument: Conformal IV Regression for Pricing Teams"
date: 2026-04-01
categories: [techniques, causal-inference]
tags: [conformal-prediction, instrumental-variables, npiv, endogeneity, telematics, prediction-intervals, uncertainty-quantification, double-machine-learning, causal-inference, insurance-conformal, insurance-causal-policy, arXiv-2603-25509, Kato, finite-sample-guarantee, python]
description: "Standard conformal prediction breaks under instrumental variable regression — the calibration residuals are not exchangeable. Kato (arXiv:2603.25509, March 2026) fixes this by redefining the coverage target. We explain what changes, why it matters for telematics and GIPP pricing, and why we are not building it yet."
math: true
author: burning-cost
---

[Our post on Panel IV DML](/pricing/causal-inference/rate-change/2026/03/26/panel-dml-instrumental-variables-when-did-isnt-enough/) ended in a familiar place: you have the causal point estimate, extracted via instrumental variables, and you want to know what surrounds it. The DML estimator gives you an average structural function $\hat{h}(X)$ — the causal effect of endogenous usage on claim cost, say, or causal demand response to price. But for a new risk at test time, the point estimate alone is not enough. You want a prediction interval. And if your estimate came from an IV model, your prediction interval needs to know that.

The natural instinct is to reach for split conformal prediction. Fit the IV model on a training set, compute residuals on a calibration set, take the appropriate quantile, use it as the half-width for new predictions. This is exactly how our [`insurance-conformal`](/insurance-conformal/) library handles standard regression. It provides finite-sample, distribution-free coverage guarantees — provided the calibration and test data are exchangeable.

That proviso is the problem. Under IV, the data is not exchangeable in the way conformal needs.

A paper from Masahiro Kato, submitted to arXiv on 26 March 2026 (arXiv:2603.25509), gives the first distribution-free prediction interval procedure for nonparametric IV regression. The result is theoretically clean and practically relevant for pricing teams that are running IV-corrected causal models — telematics endogeneity being the most immediate UK example. We are not building it yet, for reasons we will explain. But the conceptual framework is worth understanding now.

---

## The telematics selection problem

Telematics introduces a well-known endogeneity. Drivers who opt into usage-based insurance are not a random sample of all drivers. Good drivers — lower accident frequency, smoother braking, lower mileage — opt in at higher rates. When you fit a model predicting claim cost from telematics usage data, the usage signal is contaminated by this selection. The raw association between low mileage and low cost partly reflects the quality of the drivers who drove those miles, not the miles themselves.

The IV fix is standard: find a variable $Z$ that predicts whether a driver opts into telematics (relevant) but has no direct effect on claim cost beyond its effect through take-up (exclusion restriction). Postcode telematics penetration rate is a candidate: a postcode where 40% of policyholders have accepted telematics differs from one where 10% have, and the penetration rate is correlated with individual uptake but arguably not with individual driving quality.

With a valid instrument, nonparametric IV regression (NPIV) recovers $h_0(X)$ — the structural function connecting usage to cost, purged of selection bias. The model specification is:

$$Y = h_0(X) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid Z] = 0$$

where $Y$ is claim cost, $X$ is usage (endogenous), $Z$ is the instrument, and $\varepsilon$ is the structural error. The moment condition $\mathbb{E}[\varepsilon \mid Z] = 0$ is what IV exploits, and it is also what breaks standard conformal.

You now have $\hat{h}(X)$ from your NPIV estimator — kernel IV, sieve 2SLS, DeepIV, whatever you have available. For a new driver with features $X_{n+1}$ and instrument value $Z_{n+1}$, what is the 90% prediction interval?

---

## Why standard conformal breaks here

Split conformal prediction works as follows. Fit your model on a training set. On a separate calibration set of size $m$, compute nonconformity scores $S_i = |Y_i - \hat{h}(X_i)|$. For a new test point, the prediction interval is $\hat{h}(X_{n+1}) \pm q_\tau$, where $q_\tau$ is the $\lceil (1-\alpha)(m+1) \rceil / m$ quantile of the calibration scores. The coverage guarantee — $\mathbb{P}(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$ — holds under one assumption: that the calibration points and the test point are exchangeable, meaning drawn from the same distribution.

Under IV, this fails. The calibration residuals $S_i = |Y_i - \hat{h}(X_i)|$ are not exchangeable with a test residual $S_{n+1}$ when the instrument distribution at test time differs from the calibration distribution. And in any realistic deployment, it will differ. The whole point of the telematics use case is that penetration rates are shifting — that is why you need the instrument in the first place.

More precisely: the IV moment condition $\mathbb{E}[\varepsilon \mid Z] = 0$ links the structural error to the instrument. If you reweight the test population by instrument value — say you are pricing a new postcode cohort with higher telematics penetration — the conditional distribution of $Y$ given $X$ changes, because the selection process changes. The calibration quantile $q_\tau$ was computed under the old distribution. It is no longer the right radius.

There is also a deeper obstruction. Lei and Wasserman (2013) showed that no distribution-free procedure can achieve exact conditional coverage $\mathbb{P}(Y \in C \mid Z = z) = 1 - \alpha$ for all $z$ simultaneously, with finite-length intervals in general. This is an information-theoretic impossibility, not a gap in current methods. Any approach that claims exact conditional IV coverage for all possible instrument values is either wrong or making strong distributional assumptions.

Kato's paper is honest about this. Rather than overclaiming, it replaces the impossible target with the right target.

---

## The IV shift class: what coverage is actually achievable

The key conceptual move is to redefine what coverage means for an IV-estimated model. Instead of demanding exact conditional coverage for every possible instrument value, the paper asks for coverage under every distribution in a practitioner-specified class of IV reweightings.

Define a **shift class** $\mathcal{F}$ as a set of non-negative measurable functions $f: \mathcal{Z} \to \mathbb{R}_+$ on the instrument space. The coverage requirement becomes:

$$\mathbb{E}[f(Z) \cdot \mathbf{1}(Y \in C)] \geq (1 - \alpha) \cdot \mathbb{E}[f(Z)] \quad \text{for all } f \in \mathcal{F} \text{ with } \mathbb{E}[f(Z)] > 0$$

Each $f$ tilts the instrument distribution. Integrating against $f(Z)$ instead of uniform weights gives coverage under the reweighted distribution $P_f$ — the distribution you get by upweighting observations where the instrument takes high values of $f$ and downweighting where it takes low values.

For the telematics use case, $Z$ is the postcode telematics penetration rate. The shift class $\mathcal{F}$ might be the set of linear functions: $f(z) = 1 + \gamma \cdot z$ for $\gamma \in [-\gamma_{\max}, \gamma_{\max}]$. Membership in $\mathcal{F}$ says: our intervals must be valid whether penetration rates increase or decrease, up to the amount parameterised by $\gamma_{\max}$. Concretely, this means: if the industry moves from 20% telematics penetration to 40% (an upward shift in $Z$), our intervals for new risks in that environment still have 90% coverage.

The choice of $\mathcal{F}$ is the key modelling decision, and there is no data-driven default. The practitioner must specify it. This is not a weakness — it is the correct framework. An insurance pricing team deploying IV-corrected intervals should be able to state, explicitly, what instrument distribution shifts their intervals are robust to. If they cannot, the intervals are not defensible.

A larger $\mathcal{F}$ gives stronger robustness but inflates interval length — wider shifts require more conservative radii. This is exactly the right tradeoff to expose.

---

## The Z-indexed construction

The paper studies three ways to construct the interval radius. The recommended approach is the **Z-indexed class** (denoted $\mathcal{T}_Z$ in the paper), where the interval radius adapts to the instrument value but the interval centre depends only on $X$.

The construction is a variant of split conformal. Split the data into training, calibration, and test sets. On the training set, fit the NPIV estimator to obtain $\hat{h}$. Choose a shift feature map $\phi(Z) = (\phi_1(Z), \ldots, \phi_d(Z))$ — a $d$-dimensional encoding of the instrument that spans your shift class $\mathcal{F}$.

On the calibration set of size $m$, compute nonconformity scores $S_i = |Y_i - \hat{h}(X_i)|$. For a test point $(X_{n+1}, Z_{n+1})$, the prediction interval is:

$$C(X_{n+1}, Z_{n+1}) = \hat{h}(X_{n+1}) \pm q_\tau(Z_{n+1})$$

where $q_\tau(Z_{n+1})$ is the quantile of the calibration scores weighted by $\phi(Z_{n+1})^\top \phi(Z_i)$ — an inner product that upweights calibration points whose instrument values are similar to the test point.

The finite-sample coverage guarantee (Theorem 6.2 in Kato) holds exactly: for every $f \in \mathcal{F}$ with $\mathbb{E}[f(Z)] > 0$,

$$\mathbb{P}_f(Y_{n+1} \in C(X_{n+1}, Z_{n+1})) \geq 1 - \alpha$$

Distribution-free. No distributional assumptions on $Y$, $X$, $\varepsilon$, or $Z$ beyond what IV already requires.

The length inflation bound (Theorem 6.4) is also informative. The calibrated radius exceeds the oracle by at most:

$$\frac{d}{(m+1) \cdot p_{\min}} + \|\hat{h} - h_0\|_\infty$$

where $d$ is the dimension of the shift feature map, $m$ is calibration set size, and $p_{\min}$ is the minimum coverage probability over $\mathcal{F}$. The second term directly links NPIV estimation quality to interval length. A better NPIV estimator produces shorter intervals. This is the right incentive structure: invest in the base IV model, and the conformal layer rewards you with tighter intervals.

---

## Comparing the three radius classes

The paper studies three radius classes, each making a different tradeoff.

| Class | Radius depends on | Coverage type | Practical stability |
|---|---|---|---|
| $\mathcal{T}_{XZ}$ (joint-indexed) | Both $X$ and $Z$ | Exact finite-sample | Fragile: produces infinite intervals in $d \geq 3$ with RKHS |
| $\mathcal{T}_Z$ (Z-indexed) | $Z$ only | Exact finite-sample | Stable across all simulation settings |
| $\mathcal{T}_X$ (X-indexed) | $X$ only | Single-shift only | Conservative; largest interval lengths |

The joint-indexed class $\mathcal{T}_{XZ}$ is the most expressive — the radius adapts to the full conditioning information. But in the paper's simulation study on 3-dimensional data, the RKHS-based implementation produced infinite intervals. The class cannot be stably calibrated with the sample sizes available in practice. It is theoretically attractive and practically useless in most insurance settings.

The X-indexed class $\mathcal{T}_X$ produces intervals depending only on $X$, not the instrument. This would be the most natural output — a pricing model that uses instruments for causal identification but produces standard-looking prediction intervals at deployment. The problem is that it requires importance-weighted conformal calibration (density ratio estimation of the test vs calibration instrument distribution), and the coverage guarantee weakens: it holds for a single pre-specified shift distribution $f_0$, not the full family $\mathcal{F}$. That is a material weakening. An interval that is calibrated for one specific future instrument distribution but fails under others is not robust in any useful sense.

The Z-indexed class is the practical recommendation. Intervals adapt to instrument value, remain finite in all tested settings, and provide family-wise coverage across $\mathcal{F}$. In the paper's simulation (1D IV, target $1 - \alpha = 0.9$), Z-indexed coverage lands at 0.906 — exactly right. The interval length of roughly 4.0 is stable and finite.

We would use Z-indexed in any implementation.

---

## UK pricing use cases, ranked by instrument defensibility

The instrument is always the constraint. The conformal layer is software; instrument validity is economics. We rank the main UK pricing use cases by how defensible the exclusion restriction is.

**1. Inflation-instrumented claim severity (most defensible)**

Claim severity is endogenous to inflation — both the claim amount and the benchmark price index are driven by the same underlying supply chain pressures. The IV fix is a lagged supply chain index: ONS Producer Price Index lagged 3–6 months, or the Solera parts price index lagged similarly. The exclusion restriction argument is clean: the lagged index reflects cost conditions from several months ago, which have no direct path to the current individual claim other than through their effect on current repair prices. Lagged instruments are among the most credible in economics.

Application: NPIV severity model with Z-indexed conformal intervals valid under different forward inflation paths. This is directly useful for reserve adequacy: the prediction interval for outstanding claims liabilities is valid under shifts in the inflation environment — precisely the scenario a reserving actuary needs to stress-test.

**2. Telematics endogeneity (strong case; exclusion requires care)**

As described above. Postcode telematics penetration rate is the cleanest instrument candidate: correlated with individual take-up (relevant), and plausibly uncorrelated with individual driving quality beyond its effect through take-up (exclusion). The exclusion restriction requires arguing that living in a high-penetration postcode does not independently affect driving behaviour through any channel other than telematics uptake itself. This is debatable — high-penetration postcodes may differ on urbanicity, road type, or demographic characteristics that directly affect risk. Vehicle age is another candidate but weaker: it is relevant (older vehicles have lower telematics hardware rates) but almost certainly affects driving skill directly.

The Z-indexed conformal wrapper gives intervals that are valid under shifts in telematics penetration patterns — exactly what a UBI pricing team needs as the product matures and the selected population changes.

**3. GIPP price elasticity (interesting; instrument identification is hard)**

Post-General Insurance Pricing Practices (GIPP), observed price is endogenous to risk — the pricing model sets price as a function of risk characteristics, so price and risk are jointly determined. A causal demand model requires IV. Candidate instruments: competitor price indices (if firms are sufficiently differentiated that competitor pricing does not track your own risk book), FCA market study reference prices, or historical pricing model version (if the model changed for reasons unrelated to risk).

None of these are clean. Competitor prices are correlated with industry-wide risk trends. Historical pricing version requires strong assumptions about why the model changed. Consumer Duty obligations give the problem real regulatory weight — the FCA expects firms to understand their demand response — but instrument defensibility is genuinely difficult here. We would not use NPIV for elasticity estimation without a serious identification argument.

**4. Reserving development endogeneity (speculative)**

IBNR development factors are endogenous to claims handler caseload: handlers under pressure close cases faster, which affects development patterns and correlates with severity in complex ways. A potential instrument is handler caseload at claim inception — an operational variable that affects handler behaviour (relevant) but should not directly affect the underlying economic magnitude of the claim (exclusion). The exclusion restriction is plausible; the data infrastructure requirement is substantial. This use case is theoretically clean but requires linking claims system data to operational workforce data in a way most UK insurers are not currently set up for.

---

## Why we are not building this yet

The implementation verdict is straightforward: **blog, not build**. Three reasons.

First, there is no stable NPIV estimator in Python. The conformal layer wraps an existing NPIV estimator — it is estimator-agnostic. But before you can calibrate conformal IV intervals, you need $\hat{h}$ from a working NPIV fit. The Python ecosystem here is thin. EconML's IV methods (`DeepIV`, `DMLIV`) cover semiparametric IV with a linear structural function, not full nonparametric IV. `pykerneliv` exists but is sparsely maintained. A sieve 2SLS implementation from scratch would add around 600–900 lines of validated econometric code. Building conformal IV wrappers on top of an immature base layer produces brittle software.

Second, the paper is six days old. No community has stress-tested it. There is no reference Python implementation to validate against. Theorem proofs that look correct can contain edge cases that emerge under real data conditions — high collinearity between $X$ and $Z$, heavy-tailed $Y$, near-violation of the exclusion restriction. We prefer to let the method mature before committing to an implementation.

Third, the audience who needs this is currently blocked upstream. A pricing team cannot use conformal IV intervals without a valid instrument. Most UK pricing teams running telematics models have not formally validated an instrument — they are managing selection bias via observed proxies (driver score, declared mileage, modelled exposure) rather than IV. The conformal wrapper adds value only after the IV identification problem is solved. That is a more fundamental constraint than software availability.

When to reassess: watch the EconML IV roadmap for NPIV support, and watch `mliv` and `pykerneliv` for signs of active maintenance. When a mature NPIV Python library is available — one with test coverage, documentation, and a user base — a `ConformalIVPredictor` class in `insurance-conformal` is a reasonable one-week build: roughly 500 lines wrapping the external NPIV estimator, with a calibration method that implements the Z-indexed quantile weighting, and a coverage diagnostic.

---

## What the paper actually gives you

The theoretical contribution is precise and non-trivial. Here is what it actually achieves.

Before this paper, there was no distribution-free prediction interval method for NPIV. If you had an IV-estimated causal model and needed prediction intervals, your choices were parametric (assume Gaussian errors, which fails for claim costs) or bootstrap (asymptotic, computationally expensive, and invalid under the specific IV distributional shifts you care about).

After this paper, there is a principled answer. The interval centre is your NPIV point estimate. The radius adapts to the instrument value. Coverage is guaranteed — exactly, in finite samples, without distributional assumptions — under every instrument distribution in your specified shift class. The price is that you must specify the shift class, and you must accept that the coverage guarantee is over that class rather than for every possible instrument value.

That is not a weakening; it is the right framing. An insurance prediction interval that claims to be valid for every conceivable future is not credible. An interval that is valid under a specified, economically motivated class of instrument distribution shifts is both credible and useful. It gives pricing teams a language for writing down explicitly what robustness they are claiming, and a method that delivers it.

The causal/conformal intersection is where serious uncertainty quantification in pricing is heading. When your model uses IV, your intervals need to know.

---

*arXiv:2603.25509 — Kato, M. (2026). Conformal Prediction for Nonparametric Instrumental Regression. Related libraries: [insurance-conformal](/insurance-conformal/), [insurance-causal-policy](/insurance-causal-policy/). Related post: [Panel DML with Instrumental Variables](/pricing/causal-inference/rate-change/2026/03/26/panel-dml-instrumental-variables-when-did-isnt-enough/).*
