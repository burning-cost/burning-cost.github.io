---
layout: post
title: "Differential Privacy for Insurance Synthetic Data: Why DP-CTGAN Fails and What Actually Works"
date: 2026-03-31
categories: [techniques, libraries, privacy]
tags: [differential-privacy, synthetic-data, insurance-synthetic, AIM, smartnoise, GDPR, ICO, FCA, python, motor, privacy]
description: "DP-CTGAN produces near-random output at epsilon=1 on datasets under 50K rows — which is most insurance portfolios. AIM via smartnoise-synth is the correct tool. Here is the full picture: what degrades, what survives, and what to build."
---

The reason most teams reach for DP-CTGAN first is understandable. CTGAN already works for your non-private synthetic data. Wrapping it in differential privacy seems like a natural extension. Slap Opacus on the discriminator gradients, set `epsilon=1`, and you have differentially private synthetic data you can share with a vendor.

What you actually have is noise.

At epsilon=1 on any insurance dataset below roughly 50,000 rows, DP-CTGAN produces synthetic data with AUC near 0.5 in downstream classification tasks. Not "somewhat degraded" — near-random. The generator cannot learn a useful conditional distribution when DP-SGD has clipped and noised the discriminator gradients into incoherence. Training converges to the majority class. A model scored against DP-CTGAN output predicts everything as the dominant class. You have spent your privacy budget on nothing.

This is not a CTGAN-specific problem. It is the fundamental reason to use a different approach entirely.

---

## How the winning approach actually works

The empirical winner for tabular differential privacy is the **select-measure-generate** (SMG) paradigm. It does not train a neural network. Instead:

1. **Select** a collection of low-dimensional marginal queries — typically all 1-way marginals plus informative 2-way marginals
2. **Measure** those marginals privately by adding calibrated Gaussian noise (this is the DP step)
3. **Generate** synthetic data by fitting a probabilistic graphical model (PGM) to the noisy measurements and sampling from it

The DP guarantee comes entirely from step 2. Once the noisy measurements are released, you can generate as many synthetic rows as you like at zero additional privacy cost. The generated data inherits the (epsilon, delta) guarantee from the measurements.

Two algorithms implement this well:

**MST** (McKenna et al., 2021): privately measures all 1-way marginals and the set of 2-way marginals forming the maximum spanning tree weighted by approximate mutual information. Predictable, efficient, well understood.

**AIM** (McKenna et al., 2022): iteratively selects which marginals to measure based on estimated information gain. Consistently outperforms MST across benchmarks. AIM is the one to use.

Both are in [`smartnoise-synth`](https://github.com/opendp/smartnoise-sdk) (v1.0.6, Feb 2026), actively maintained by the OpenDP project.

---

## The benchmark numbers

The most rigorous comparison is PMC:10843030 (PLOS One, 2024), testing epsilon ∈ {0.5, 1.0, 5.0, 10.0} across multiple tabular datasets:

| epsilon | AIM AUC | MST AUC | DP-CTGAN AUC | Real data AUC |
|---------|---------|---------|--------------|---------------|
| 0.5     | ~0.62   | ~0.61   | ~0.50        | 0.684         |
| 1.0     | ~0.65   | ~0.64   | ~0.50        | 0.684         |
| 5.0     | 0.683   | 0.662   | ~0.50        | 0.684         |
| 10.0    | 0.684   | 0.662   | ~0.51        | 0.684         |

DP-CTGAN flatlines. AIM and MST are meaningfully below real-data quality at epsilon=1, but functional — around 0.65 vs 0.684. Above epsilon=5, AIM essentially matches real data.

The diffusion-based alternative — DP-FinDiff (arXiv:2512.00638, Dec 2024) — achieves AUC=0.768 at epsilon=1, a genuine improvement over DP-CTGAN's 0.515. But there is no production Python package for it yet. It is research code. When it reaches a stable release, it becomes interesting. For now, AIM is what you can actually use.

---

## What epsilon means in practice, and why nobody can tell you the right number

(epsilon, delta)-DP means: for any two datasets differing by one record, and any output set S, the probability ratio P[M(D) ∈ S] / P[M(D') ∈ S] is at most e^epsilon. delta is a failure probability, typically 1/n or 1e-5.

In plain terms: the adversary's ability to determine whether your specific policyholder's record contributed to the synthetic dataset is bounded by e^epsilon. At epsilon=1, that multiplier is e ≈ 2.7. At epsilon=10, it is e^10 ≈ 22,000. The guarantee degrades dramatically at high epsilon.

Membership inference attack results confirm this directly (arXiv:2402.06699):

| epsilon | MST adversary advantage | PrivBayes adversary advantage |
|---------|------------------------|-------------------------------|
| 1       | 0.56                   | 0.53                          |
| 10      | 0.72                   | 0.64                          |
| 100     | 0.77                   | 0.88                          |
| 1,000   | 0.77                   | 0.96                          |

At epsilon=1 the adversary advantage is 0.53–0.56 — barely above the random baseline of 0.5. At epsilon=10 they have a meaningful edge. Above epsilon=100 the formal guarantee is effectively hollow.

**What the ICO says.** The ICO's updated anonymisation guidance (March 2025) does not specify an epsilon threshold. Their test is contextual: would a determined attacker with reasonable resources likely succeed in re-identifying the data? DP is explicitly listed as a technique that *can* support anonymisation, but "synthetic data may or may not be anonymous" — assessed case by case. There is no formula. No epsilon floor exists in ICO guidance, FCA guidance, or the Data Use and Access Act 2025 (which received Royal Assent 19 June 2025).

Our practical view: epsilon=1–3 with a documented membership inference audit (showing adversary advantage below ~0.6) provides a defensible evidence base for the ICO's motivated-intruder test when sharing externally. For internal model development, you do not need DP at all — the existing vine copula approach in [`insurance-synthetic`](/insurance-synthetic/) is the right tool.

---

## What DP destroys, and what survives

This is the honest part. The degradation is a mathematical consequence of the DP mechanism, not a tuning problem.

**What degrades first as epsilon decreases:**

1. **Rare class preservation.** Fraud indicators, flood claims, critical illness events are typically 0.5–2% of records. A 1% class in a 10,000-policy dataset is 100 records. At epsilon=1, the DP noise floor makes that class proportion statistically indistinguishable from zero. AIM will not preserve fraud flags at epsilon=1 unless your dataset has at least 50,000–100,000 policies. State this limit clearly, do not paper over it.

2. **Tail quantiles.** TVaR at P99 is degraded 20–40% at epsilon=1 even with AIM/MST. The mechanism discretises continuous columns into bins; the high-value bins (large claims) are sparse, and DP noise applied to a sparse bin with count ~50 is proportionally enormous. A bin containing 50 claims at epsilon=1 with n=100K policies gets Gaussian noise with sigma ~50. The synthetic tail count is random.

3. **High-cardinality interactions.** Multi-way interactions — young driver × sports car × London postcode — get deprioritised as AIM selects which marginals to measure. Only the statistically dominant pairwise relationships survive at epsilon=1.

**What survives reasonably well:**

- Marginal distributions of main rating factors: largely preserved at epsilon=3+, TVD below 0.10 for most columns above epsilon=5
- Pairwise correlations: AIM achieves Wasserstein fidelity around 0.15–0.20 at epsilon=5
- GLM main effects: coefficients within ~25% of non-DP synthetic at epsilon=1. Interaction terms: not preserved.

The tail problem is fundamental. The data that matters most for insurance pricing — large claims, rare perils, niche segments — is precisely what DP protects most aggressively. You cannot simultaneously protect a claimant with a £300K loss (rare, identifying) and preserve the Pareto tail parameter that drives 99th-percentile pricing loads. These goals are in direct tension. The solution is a hybrid approach: DP for the bulk distribution, a separate parametric tail model fitted with formal DP marginal queries on the exceedance population. This is not a cop-out; it is the correct actuarial design.

---

## The domain extraction trap

There is a subtle failure mode that breaks end-to-end DP guarantees in every standard library, including smartnoise-synth by default (arXiv:2504.06923).

When you call `synthesizer.fit(data, epsilon=1.0)`, the library needs to know the domain of each column: min, max for continuous columns, the set of categories for categorical columns. By default, it fits this from the data non-privately. This domain extraction step is not covered by the epsilon budget you specified.

The consequence: even at epsilon=1, non-private domain extraction enables 100% membership inference success on outlier records — the £280K claim, the 17-year-old with a modified vehicle. The adversary simply asks: "is this record in the dataset? It must be if it's near the min or max." The formal DP guarantee you thought you had is broken for exactly the records you most need to protect.

The fix is straightforward and costs nothing: pre-specify column bounds from actuarial knowledge.

```python
column_bounds = {
    "driver_age": (17, 100),        # statutory — every actuary knows this
    "vehicle_group": None,           # categorical, known levels
    "ncd_years": None,               # ordinal categorical, 0–15
    "vehicle_value": (500, 150_000), # known from underwriting guidelines
    "claim_amount": (0, 500_000),    # calibrated P99 threshold; tail handled separately
    "exposure": (0.0, 1.0),         # fractional policy years
}
```

These bounds are actuarial knowledge, not data. Using them costs zero epsilon and eliminates the domain extraction vulnerability.

---

## The insurance-specific problems no DP tool solves

Standard DP synthetic data benchmarks use health records, credit data, census data. These do not have the structural properties that make insurance data difficult.

**Exposure and frequency.** Insurance claim frequency is Poisson(λ × exposure). Exposure is not a feature to synthesise — it is a fixed attribute of each policy that encodes time-at-risk. If AIM synthesises exposure alongside rating factors, it will destroy the Poisson structure. A synthetic policy with 0.1 years' exposure will not have claim counts drawn from a distribution with 10% of the expected value for a full-year policy.

The correct approach is to handle exposure separately: estimate the claim rate λ from a private count query on the real data (adding Gaussian noise to both the numerator and denominator), generate claim counts as Poisson(λ_dp × exposure_i), and either resample real exposures or draw from a fitted Beta distribution. AIM does not do this natively. No DP tool does.

**Frequency-severity separation.** The zero-inflated structure of claim amounts — 85–90% zeros, remainder right-skewed — must be handled by separate DP models for claimant selection (Bernoulli) and conditional severity. Running the zero-inflated column through AIM as a single continuous column produces garbage: the discretisation bins span zero through £500K, and DP noise makes the zero-bin proportion uncertain.

**Postcode granularity.** UK motor has approximately 9,500 postcode sectors. AIM cannot handle a 9,500-category column at epsilon=1. Collapse to postcode area (124 areas) before synthesis and publish the mapping as non-private metadata. You lose sub-area geographic variation, but you retain the pricing signal that matters.

---

## The code: smartnoise-synth AIM in practice

Install:

```bash
uv add smartnoise-synth
# AIM also requires private-pgm (not on PyPI):
uv add git+https://github.com/ryan112358/private-pgm.git
```

Minimal working example:

```python
import pandas as pd
from snsynth import Synthesizer

# Load your (preprocessed, discretised) policy data
df = pd.read_parquet("motor_policies.parquet")

# Pre-specify bounds to avoid domain extraction vulnerability
# Continuous columns must be discretised before fitting
# (use PrivTree discretisation for best results at epsilon=1)
df["driver_age_bin"] = pd.cut(df["driver_age"], bins=list(range(17, 101, 5)))
df["vehicle_value_bin"] = pd.cut(
    df["vehicle_value"],
    bins=[0, 5000, 10000, 20000, 40000, 80000, 150000]
)

# Drop columns to be handled separately
features = df.drop(columns=["claim_amount", "claim_count", "exposure"])

# Fit AIM synthesizer
synth = Synthesizer.create("aim", epsilon=1.0, verbose=True)
synth.fit(
    features,
    preprocessor_eps=0.1,  # 10% of budget for PrivTree discretisation
)

# Generate synthetic policies
synthetic_features = synth.sample(len(df))

# --- Handle frequency-exposure separately ---
import numpy as np

# Estimate lambda from DP count query — Laplace mechanism (pure epsilon-DP, sensitivity=1)
# Correct noise scale for Laplace mechanism is sensitivity / epsilon = 1 / epsilon_lambda.
# Do not use np.random.normal here: Gaussian mechanism requires a larger sigma of
# sqrt(2 * ln(1.25/delta)) / epsilon and provides (epsilon, delta)-DP, not pure DP.
epsilon_lambda = 0.2  # budget for this query (from remaining budget)
n_claims_noisy = df["claim_count"].sum() + np.random.laplace(0, 1.0 / epsilon_lambda)
total_exposure_noisy = df["exposure"].sum() + np.random.laplace(0, 1.0 / epsilon_lambda)
lambda_dp = max(0, n_claims_noisy / total_exposure_noisy)

# Generate exposures (resample from real distribution — treat as non-sensitive)
synthetic_features["exposure"] = np.random.choice(df["exposure"].values, size=len(df))
synthetic_features["claim_count"] = np.random.poisson(
    lambda_dp * synthetic_features["exposure"]
)
```

The `preprocessor_eps=0.1` parameter allocates 10% of your epsilon to PrivTree discretisation of any remaining continuous columns. The 90% remainder goes to AIM's marginal measurements.

For the severity column:

```python
# Fit a separate AIM synthesizer on non-zero claims only
claims_df = df[df["claim_amount"] > 0][["claim_amount", "vehicle_group", "driver_age_bin"]]
claims_df["claim_amount_bin"] = pd.cut(
    claims_df["claim_amount"],
    bins=[0, 1000, 3000, 7000, 15000, 35000, 100000, 500000],
    labels=False,
)
claims_df = claims_df.drop(columns=["claim_amount"])

severity_synth = Synthesizer.create("aim", epsilon=0.5)  # separate budget allocation
severity_synth.fit(claims_df, preprocessor_eps=0.05)
```

Running AIM on a 50K-row UK motor dataset at epsilon=1 takes 2–5 minutes on a standard laptop. The smartnoise-synth API handles Rényi DP accounting automatically — you pass epsilon and it computes the Gaussian noise scale correctly.

---

## What we recommend building

The gap in the current tooling is a thin wrapper that handles the insurance-specific structure — exposure, frequency-severity separation, pre-specified column bounds — around smartnoise-synth's AIM synthesizer. We estimate around 500 lines in [`insurance-synthetic`](/insurance-synthetic/), as an optional module with smartnoise-synth as an optional dependency.

The class design we have in mind:

```python
class DPInsuranceSynthesizer:
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        mechanism: str = "aim",      # "aim" or "mst"
        column_bounds: dict | None = None,
        severity_epsilon_fraction: float = 0.3,
    ):
        ...

    def fit(
        self,
        data: pl.DataFrame,
        exposure_col: str = "exposure",
        claim_count_col: str = "claim_count",
        severity_col: str | None = None,
    ) -> "DPInsuranceSynthesizer":
        # 1. Validate column_bounds cover all continuous columns
        # 2. Allocate epsilon: 10% discretisation, 60% policy features, 30% severity
        # 3. Fit AIM on policy features (no exposure, no severity)
        # 4. Fit separate DP marginal on non-zero severity
        # 5. Estimate Poisson lambda via DP count query
        ...

    def generate(self, n: int) -> pl.DataFrame:
        ...

    def privacy_report(self) -> DPPrivacyReport:
        # epsilon_spent, delta, mechanism, membership inference audit
        ...
```

Helper functions providing pre-specified bounds remove the domain extraction risk without requiring actuarial input every time:

```python
def uk_motor_column_bounds() -> dict:
    return {
        "driver_age":     (17, 100),
        "vehicle_group":  None,           # categorical
        "ncd_years":      None,           # categorical
        "vehicle_value":  (500, 150_000),
        "claim_amount":   (0, 500_000),   # non-zero claims only
        "exposure":       (0.0, 1.0),
    }
```

What a v1 release should explicitly document as limitations, not paper over:

- Claim severity tail (P99+) is degraded 20–40% at epsilon=1. Use a separately calibrated Pareto tail model for extreme claims.
- Rare events (fraud <2%) are not preserved at epsilon=1 for datasets under 50K rows. Document the minimum n for each use case.
- GLM interaction terms within ±25% at epsilon=1 is not guaranteed.
- epsilon=1 is not a regulatory guarantee of anonymisation under ICO guidance. A membership inference audit is required alongside the epsilon declaration.

What we are not building in v1:

- **DP-CTGAN**: the benchmark evidence is unambiguous — categorically worse than AIM for insurance dataset sizes. Not worth the engineering time.
- **DP vine copula from scratch**: architecturally the cleanest fit for insurance-synthetic's existing design, but would require reimplementing COPULA-SHIRLEY (Gambs et al., PoPETs 2021) from scratch in Python 3.12 against pyvinecopulib. Estimated 3–4× development effort for marginal utility gain over AIM. Deferred to v2.
- **DP-FinDiff**: best utility at epsilon=1 from the literature (arXiv:2512.00638), but no production code exists. Track the repo.

---

## The regulatory picture

Neither the ICO nor the FCA mandates differential privacy for synthetic insurance data. The primary regulatory driver for DP is external data sharing — releasing data to vendors, research institutions, or open publication — not internal model development.

The ICO's motivated-intruder test asks whether a determined attacker with reasonable resources would succeed in re-identifying individuals. Epsilon=1–3 combined with a documented membership inference audit (adversary advantage below ~0.6) is a defensible answer to that test. There is no required epsilon value.

The FCA's Synthetic Data Expert Group report (August 2025) requires firms to demonstrate privacy risk assessment, auditability, and bias management during synthesis. A DP approach with documented epsilon and a membership inference audit satisfies this clearly. But DP is not required even for external sharing — it is a governance improvement, not a compliance floor.

For internal use: stick with the existing [`insurance-synthetic`](/insurance-synthetic/) vine copula approach. It is faster, preserves tails better, and produces no spurious privacy claims.

---

## The honest summary

DP-CTGAN is not useful for insurance synthetic data at epsilon=1. The benchmarks on this are consistent across multiple independent studies. The mechanism is simply unable to learn from enough signal when gradient clipping and noise dominate the training signal at typical insurance dataset sizes.

AIM via smartnoise-synth is the correct choice. It is functional at epsilon=1 on datasets above ~10,000 rows, approaches real-data quality above epsilon=5, and handles typical UK motor schema columns within documented limitations.

The limitations are real and specific: tail quantiles degrade 20–40% at epsilon=1, rare events below 2% are not preserved below 50K rows, and the frequency-exposure relationship requires explicit handling that no existing DP tool provides. These are not bugs to fix. They are the mathematical consequences of meaningful privacy guarantees applied to the data that is most informative and most sensitive.

The engineering task — a ~500-line wrapper around AIM with insurance-specific structure — is well-defined. We will write it up when we have benchmark results on a UK motor schema.

---

*The benchmarks in this post reference PMC:10843030 (PLOS One, 2024), arXiv:2504.06923, arXiv:2512.00638, arXiv:2402.06699, and the ICO anonymisation guidance (March 2025). The insurance-synthetic library is at [github.com/burning-cost/insurance-synthetic](https://github.com/burning-cost/insurance-synthetic).*
