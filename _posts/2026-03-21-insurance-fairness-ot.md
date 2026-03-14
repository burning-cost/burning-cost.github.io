---
layout: post
title: "Optimal Transport for Discrimination-Free Insurance Pricing"
date: 2026-03-21
categories: [libraries, pricing, fairness]
tags: [fairness, discrimination, optimal-transport, Wasserstein, Lindholm, marginalisation, FCA, EP25/2, consumer-duty, proxy-discrimination, insurance-fairness-ot, python, motor]
description: "The FCA's EP25/2 consultation set out a coherent framework for discrimination-free pricing. The underlying mathematics — Lindholm marginalisation and Wasserstein barycenters — had no open-source Python implementation. insurance-fairness-ot fixes that."
canonical_url: "https://burning-cost.github.io/2026/03/10/insurance-fairness-ot/"
---

The FCA's CP23/22 and subsequent EP25/2 established that insurers must be able to demonstrate their pricing does not unfairly discriminate against protected characteristics — and that proxy discrimination through correlated variables counts. The regulatory intent is clear. The technical implementation is not.

The standard response from pricing teams is a variant of: remove the protected characteristic, check that your model's accuracy does not change much, write a paragraph in the model validation report. This is not what the regulation requires. It checks whether the model uses the characteristic directly; it does not check whether the model produces systematically different prices for identically risky people who differ only in their protected characteristic status.

The correct technical framework is well-established in the academic literature: Lindholm, Richman, Tsanakas, and Wüthrich (2022) provide the mathematical foundation in "Discrimination-Free Insurance Pricing" (ASTIN Bulletin). The core insight is that proxy discrimination is not about model inputs — it is about the causal structure between features, protected characteristics, and outcomes. Removing postcode does not remove gender discrimination if postcode correlates with gender and has a causal path through gender to claim frequency.

[`insurance-fairness-ot`](https://github.com/burning-cost/insurance-fairness-ot) implements the Lindholm et al. framework plus Wasserstein barycenter pricing: three methods for producing discrimination-free prices, with FCA EP25/2-aligned documentation output. 145 tests, MIT-licensed.

```bash
uv add insurance-fairness-ot
```

---

## What discrimination-free pricing actually means

Lindholm et al. define a premium as discrimination-free if it is equal to the expected cost conditional on risk-relevant covariates, integrated over the distribution of the protected characteristic as if it were independent of those covariates. Concretely:

For a binary protected characteristic $S$ (e.g. gender) and risk features $X$:

$$\pi^{DF}(x) = \mathbb{E}[\mu(X, S) \mid X = x]$$

where the expectation is taken over the marginal distribution of $S$ — not the conditional distribution given $X$. This removes the premium variation attributable to $S$ while preserving the variation attributable to $X$.

The practical implementation requires constructing the marginal distribution. There are three approaches in the literature, all implemented in the library:

**Lindholm marginalisation**: directly average the model's predictions over the population distribution of $S$ at each covariate profile. Requires knowing or estimating $P(S)$. Simple and interpretable; assumes $S$ is observed.

**Causal path decomposition**: uses a structural causal model to distinguish direct effects (feature → claim, causal) from proxy effects (feature → claim, mediated by $S$). Removes only the proxy path. More surgically correct but requires specifying the causal graph, which involves judgment.

**Wasserstein barycenter**: finds the price distribution that is the geometric mean (in Wasserstein distance) of the price distributions across $S$ groups. Produces a single price schedule that is equidistant from all group-conditional price schedules. No causal structure required; treats the problem as a distributional averaging problem.

---

## Usage

```python
from insurance_fairness_ot import (
    LindholmMarginaliser,
    CausalPathDecomposer,
    WassersteinBarycenterPricer,
    FairnessDiagnostics,
    FCAEvidencePack
)

# Method 1: Lindholm marginalisation
marginaliser = LindholmMarginaliser(
    model=fitted_glm,
    protected_col='gender',
    population_dist={'M': 0.52, 'F': 0.48}
)
df['df_premium'] = marginaliser.transform(df)

# Method 3: Wasserstein barycenter
wbc = WassersteinBarycenterPricer(protected_col='gender')
wbc.fit(df, premium_col='base_premium')
df['wbc_premium'] = wbc.transform(df)

# Diagnostics: does the adjusted price still discriminate?
diagnostics = FairnessDiagnostics(protected_col='gender')
report = diagnostics.evaluate(df, premium_col='df_premium', cost_col='actual_cost')
# Returns: Gini by group, mean premium ratio, calibration by group
```

---

## The evidence pack

The FCA EP25/2 framework requires documented evidence of the analysis, not just the outcome. The `FCAEvidencePack` class generates a structured HTML and JSON artefact that includes:

- Unadjusted premium distributions by protected group
- Discrimination metrics before and after adjustment (mean ratio, variance ratio, Wasserstein distance)
- Calibration tests by protected group (is the adjusted premium actuarially sound for each group?)
- Causal graph specification (if causal path decomposition was used)
- Methodology statement referencing Lindholm et al. (2022) and the EP25/2 consultation

```python
pack = FCAEvidencePack(method='lindholm', reference_date='2026-03-01')
pack.generate(df, output_path='./fairness_evidence_pack.html')
```

The output is not a model card — it is structured around the four questions in EP25/2: (1) does the model use protected characteristics directly?, (2) does it use proxies that produce disparate outcomes?, (3) if so, is that disparity causally justified?, (4) if not, what adjustment has been made?

---

## Why optimal transport

The Wasserstein barycenter approach uses optimal transport because it produces a price distribution that minimises the total "work" required to move from each group's distribution to the fair price. This has a practical implication: it minimises the total premium change required to achieve fairness, which matters because large premium shifts create anti-selection risk.

A naively averaged price (arithmetic mean of group premiums) can produce a schedule that over-prices some segments and under-prices others in ways that anti-select badly. The Wasserstein barycenter distributes the adjustment more evenly across the risk profile, preserving actuarial soundness.

---

## The calibration constraint

There is a tension between discrimination-freedom and calibration. A model that is calibrated for the population as a whole may be systematically under- or over-priced for one protected group after adjustment. The library enforces a calibration check: after any adjustment, it tests that the expected loss ratio does not materially differ across groups. If it does, the adjustment has introduced systematic mispricing and the warning is surfaced in the evidence pack.

This is not a theoretical concern. In thin segments — young male drivers, for example, who are both the primary telematics target and the most scrutinised gender group — the calibration and fairness constraints can genuinely conflict. The library does not resolve that conflict for you; it makes it visible.

---

## Relationship to insurance-fairness

The existing [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) library handles proxy discrimination detection and audit documentation: it tells you whether your model discriminates. `insurance-fairness-ot` handles the adjustment: it produces discrimination-free prices. The two libraries are designed to be used in sequence — detect with `insurance-fairness`, adjust with `insurance-fairness-ot`.

---

**[insurance-fairness-ot on GitHub](https://github.com/burning-cost/insurance-fairness-ot)** — 145 tests, MIT-licensed, PyPI.

---

**Related articles from Burning Cost:**
- [Your Pricing Model Is Discriminating. Here's Which Factor Is Doing It.](/2026/03/10/insurance-fairness-diag/)
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [Causal Mediation Analysis for Insurance Pricing](/2026/03/11/insurance-mediation/)
