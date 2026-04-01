---
layout: post
title: "Lindholm Fairness Is Not Enough for Income Protection Pricing"
date: 2026-04-01
categories: [fairness, insurance-pricing]
tags: [fairness, income-protection, critical-illness, ltci, multi-state, markov, lindholm, poisson, kolmogorov, fca, ms24-1]
description: "Standard discrimination-free pricing applies to a single outcome. Income protection premiums are derived from a matrix of transition rates. Applying Lindholm to the aggregate premium is structurally wrong. The correction must happen per transition, before you touch the Kolmogorov equations."
math: true
author: Burning Cost
---

There is a version of fairness compliance that pricing actuaries fall into by habit. You have a Lindholm-corrected model. You've marginalised out the sensitive attribute. You call this discrimination-free pricing and move on. For a single-period product — annual motor, annual home, annual travel — this is correct. For income protection, critical illness, or long-term care, it is structurally wrong, and a February 2026 paper from Lim, Xu, and Zhou (arXiv:2602.04791) explains precisely why.

The short version: an IP or LTC premium is not a single predicted value. It is a function of a matrix of transition intensities fed through the Kolmogorov forward equations. Marginalisation does not commute with the matrix exponential. If you correct the aggregate premium rather than the underlying transition rates, you are not implementing discrimination-free pricing — you are implementing an approximation that may leave substantial proxy discrimination intact.

v1.1.0 of [insurance-fairness](https://github.com/burning-cost/insurance-fairness) adds `MultiStateTransitionFairness` to address this. This post explains the theory, the implementation, and why the FCA's pure protection market study (MS24/1) makes this urgent now.

---

## The problem with single-period fairness tools on multi-state products

Lindholm et al. (ASTIN Bulletin, 2022) established the discrimination-free pricing principle for insurance: given a model $\hat{f}(z, s)$ trained on features $z$ and sensitive attribute $s$, the discrimination-free price is:

$$\hat{f}^*(z) = \int \hat{f}(z, s) \, dP_n(s)$$

This is marginalisation over the empirical distribution of $s$. It works cleanly when the premium is the model's direct output: a single predicted frequency, a single predicted severity, or their product.

Income protection is different. The premium for an IP policy is not a model output. It is the expected present value of benefits, computed by solving the Kolmogorov forward equations for a continuous-time Markov chain across states (Healthy, Disabled, Dead — or a more granular set). The premium depends on the full matrix of transition intensities $Q = \{\lambda_{ij}\}$ through the matrix exponential $e^{Qt}$. This is a nonlinear transformation of the transition rates.

Nonlinear transformations break the commutativity of marginalisation. If $\pi(\{\lambda_{ij}\})$ is the premium functional, then:

$$\mathbb{E}_s\bigl[\pi(\hat{\lambda}(z, s))\bigr] \neq \pi\bigl(\mathbb{E}_s[\hat{\lambda}(z, s)]\bigr)$$

The left side is "marginalise the premium" — apply Lindholm once to the aggregate output. The right side is "marginalise the rates, then compute the premium" — apply Lindholm per transition and rerun Kolmogorov. These are not equal. The right side is discrimination-free pricing. The left side is a different calculation that may close some of the gap by accident, but provides no formal guarantee and no audit trail that stands up to scrutiny.

The correct approach, which Lim/Xu/Zhou establish formally: apply the discrimination-free correction to each transition rate $\lambda_{ij}$ independently, then recompute the premium from the corrected rates. This is post-processing Lindholm marginalisation applied $M$ times — once per transition type — followed by a standard Kolmogorov calculation.

---

## How the paper gets there: Poisson equivalence as the bridge

The technical mechanism that makes this tractable is the Poisson equivalence. For a continuous-time Markov chain with constant hazard $\lambda_m$ over a short age interval, the number of observed transitions follows $\text{Poisson}(\tau \cdot \lambda_m)$, where $\tau \in [0,1]$ is the person-time exposure. This is standard biostatistics — it has been used in mortality modelling for decades. The contribution in arXiv:2602.04791 is using it as a bridge into the fairness literature.

If each transition $m$ in your multi-state model is a Poisson GLM with log-link and $\log(\tau)$ offset, then any fairness method that applies to a Poisson GLM applies directly to each transition. The multi-state problem reduces to $M$ independent single-period problems. Lindholm marginalisation, optimal transport pre-processing, adversarial in-processing — all of them now have a well-defined action at the transition level.

The discrimination-free transition rate formula is:

$$\lambda^*_m(z, x) = \sum_j \hat{\lambda}_m(z, x, s_j) \cdot \hat{P}(S = s_j)$$

This is the Lindholm formula applied per transition $m$. The corrected rates $\lambda^*_m$ are then assembled into the $Q$-matrix and passed through `scipy.linalg.expm` to compute the fair premium.

---

## What we built: `MultiStateTransitionFairness` in insurance-fairness v1.1.0

The module adds five classes to `insurance_fairness.multi_state`:

- `TransitionDataBuilder` — converts panel data (one row per individual-wave) to transition-level records with event counts and person-time exposure, including interval splitting at age boundaries
- `PoissonTransitionFitter` — fits $M$ independent Poisson GLMs via scipy, one per transition type, with log-exposure offset
- `MultiStateTransitionFairness` — the orchestrator: builds data, fits transitions, calls `LindholmCorrector` once per transition, computes the fair premium
- `KolmogorovPremiumCalculator` — assembles the $Q$-matrix from corrected rates and computes expected present value via matrix exponential
- `MultiStateFairnessReport` — dataclass with before/after premiums by group, per-transition rate ratios, and deviance statistics

The implementation is 1,070 lines with 42 tests. The `LindholmCorrector` from existing `optimal_transport/correction.py` handles the marginalisation — `MultiStateTransitionFairness` calls it $M$ times, passing each fitted transition model as the `model_fn` callable. No new marginalisation logic was needed.

Here is the full pipeline for a 3-state income protection model:

```python
import polars as pl
from insurance_fairness.multi_state import (
    MultiStateTransitionFairness,
    KolmogorovPremiumCalculator,
)

# panel_df: one row per individual-wave
# Required columns: id, wave, state, age, plus feature columns
# States: "H" (Healthy), "D" (Disabled), "Dead"
panel_df = pl.read_parquet("ip_panel.parquet")

# Configure the multi-state fairness corrector
mstf = MultiStateTransitionFairness(
    states=["H", "D", "Dead"],
    absorbing_states=["Dead"],
    sensitive_col="occupation_class",   # IP proxy for protected characteristics
    feature_cols=["age_band", "sex", "smoker", "benefit_amount"],
    discount_rate=0.03,
)

# Fit and transform: returns MultiStateFairnessReport
report = mstf.fit_transform(
    panel_df=panel_df,
    benefit_state="D",          # benefit paid while Disabled
    benefit_type="annuity",     # periodic payment
    term_years=20,
)

print(report)
# MultiStateFairnessReport
#   sensitive_attr: occupation_class
#   n_transitions: 4  (H→D, H→Dead, D→H, D→Dead)
#
#   Premium before correction (by occupation class):
#     Class 1 (professional):  £2,847
#     Class 2 (clerical):      £3,104
#     Class 3 (manual):        £4,219
#     Class 4 (heavy manual):  £5,831
#
#   Premium after correction (discrimination-free):
#     All groups:              £3,562
#
#   Per-transition rate ratios (Class 4 / Class 1, before → after):
#     H→D:   1.84 → 1.02
#     H→Dead: 1.31 → 1.09
#     D→H:   0.71 → 0.98
#     D→Dead: 1.22 → 1.05
#
#   Deviance before: 4,821.3  |  Deviance after: 4,904.7

# Access the corrected transition rates directly
lambda_star = report.corrected_rates  # dict[str, np.ndarray] keyed by "H→D" etc.

# Recompute premium at different discount rates
calc = KolmogorovPremiumCalculator(
    states=["H", "D", "Dead"],
    absorbing_states=["Dead"],
    benefit_state="D",
    benefit_type="annuity",
    discount_rate=0.05,
)
premium_5pct = calc.compute(lambda_star, term_years=20)
```

The `TransitionDataBuilder` and `PoissonTransitionFitter` are accessible directly if you want to inspect the intermediate representations:

```python
from insurance_fairness.multi_state import (
    TransitionDataBuilder,
    PoissonTransitionFitter,
)

# Build transition-level records
builder = TransitionDataBuilder(
    states=["H", "D", "Dead"],
    absorbing_states=["Dead"],
)
transition_df = builder.build(panel_df)
# Returns polars DataFrame: [id, from_state, to_state, age_band, event, exposure]

# Fit one Poisson GLM per transition
fitter = PoissonTransitionFitter(
    feature_cols=["age_band", "sex", "smoker", "benefit_amount", "occupation_class"],
)
fitted_models = fitter.fit(transition_df)
# Returns dict[str, fitted_params] keyed by "H→D", "H→Dead", "D→H", "D→Dead"

# Inspect predictions for a specific transition
h_to_d_rates = fitter.predict("H→D", transition_df)
```

---

## The UK regulatory context: FCA MS24/1

The FCA's pure protection market study (MS24/1) is the clearest regulatory pressure point here. The interim report landed in January 2026; the final report is expected in Q3 2026. The study covers income protection and critical illness systematically, and the direction of travel is explicit: the FCA wants to understand whether pricing practices in protection insurance produce fair outcomes across protected characteristic groups.

The relevant proxy in IP pricing is occupation class. Occupation class is a legitimate risk factor for disability inception — a coal miner has a materially different disability risk profile from a software engineer, and this is reflected in pricing. But occupation also correlates with race, gender, and disability status in ways that can produce indirect discrimination under Equality Act 2010 s.29. An IP pricing model that includes occupation as a feature will, to some degree, encode protected characteristic information through that proxy.

The per-transition Lindholm correction removes the component of each transition rate that is attributable to the sensitive attribute distribution. A model trained on `[age, sex, smoker, occupation_class]` with `occupation_class` specified as the sensitive attribute will produce corrected rates $\lambda^*_{H \to D}(z)$ that reflect risk variation across age, sex, and smoking status but not across the distribution of occupation classes. This is not the same as removing occupation from the model — the risk information in occupation is still used to fit the model. The correction adjusts the premium so that it does not vary by group in a way that tracks the sensitive attribute distribution.

For EqA compliance documentation, the `MultiStateFairnessReport` provides the audit trail: per-transition rate ratios before and after correction, deviance statistics, and before/after premiums by group. This is what a fair value assessment under Consumer Duty actually needs to show.

Critical illness sits slightly differently. The multi-state model for CI with a mental health trigger condition — Healthy → CI (including severe MH episode) — embeds the same proxy structure. If occupation or postcode appears in the CI inception model and correlates with ethnicity or disability status, the proxy discrimination operates at the transition level. The per-transition correction applies identically.

---

## What the correction does and doesn't give you

The `MultiStateFairnessReport` records deviance before and after correction. In the example above, deviance increases from 4,821 to 4,905 after correction — about a 1.7% degradation in model fit. This is the accuracy-fairness trade-off made explicit and quantified. For regulatory defence, having this number documented is preferable to the alternative, which is an implicit trade-off you cannot measure.

The correction implements demographic parity: premiums do not vary by sensitive group after marginalisation. It does not implement equalised odds or predictive parity across groups — the paper defines these but does not provide correction procedures for them. For most UK protection pricing use cases, demographic parity is the relevant standard under EqA s.29 (indirect discrimination in service provision), so this is not a material limitation.

There are edge cases worth knowing about. Sparse subpopulations — small "Other" ethnicity groups in a UK IP portfolio, for example — produce unstable empirical distributions $\hat{P}(S = s_j)$, and the marginalised rates inherit that instability. The underlying `LindholmCorrector` handles this through its `bias_correction='kl'` option, which applies a KL-optimal correction when group sizes are small. The `MultiStateTransitionFairness` exposes this via the `bias_correction` parameter.

The framework also assumes the sensitive attribute $S$ is fixed at policy issue. For disability insurance, disability is itself a protected characteristic under EqA s.6, and it is also a model state. How to handle an attribute that is simultaneously a protected characteristic and a state transition target is not resolved in the paper or in v1.1.0. We have a note in the backlog.

---

## What is not in v1.1.0

The paper also proposes age-conditional optimal transport pre-processing and adversarial in-processing. Neither is in v1.1.0.

Age-conditional OT is the paper's most technically novel contribution. Age plays a dual role in multi-state products: it drives transition rates (older individuals have higher disability inception) and enters the premium calculation directly (older policyholders have shorter remaining lifetimes over which to collect benefit). Applying unconditional OT to the feature space would scramble this age-premium relationship. The solution is a separate transport map per age cohort — actuarially reasonable, computationally straightforward, but a materially different implementation from the existing `WassersteinCorrector`. We plan this for a v1.2 PR.

Adversarial in-processing requires torch. We are not adding torch as a dependency to insurance-fairness without strong demand from actual users. If you are training per-transition adversarial encoders in production, get in touch — we want to hear about it.

---

## Install

```bash
pip install insurance-fairness==1.1.0
```

The module requires polars, numpy, and scipy — no additional dependencies beyond what insurance-fairness already carries.

Full documentation and a worked example using synthetic panel data in the [multi-state notebook](https://github.com/burning-cost/insurance-fairness/blob/main/notebooks/multi_state_income_protection.ipynb). The synthetic data matches the structure of ELSA Wave 9 (UKDS SN 5765), which is the closest public UK equivalent to the HRS data used in the paper's case study.

---

The core insight from Lim/Xu/Zhou is simple enough to state plainly: in a multi-state model, the premium is a nonlinear function of transition rates, and fairness corrections must be applied where the discrimination actually lives — in the transition rates — not on the output of a nonlinear transformation of them. If you are pricing IP or CI today and thinking about fairness compliance, the question to ask is whether your current approach corrects at the right level. For most teams, the honest answer is that it does not, because the tooling to do it correctly has not existed until now.
