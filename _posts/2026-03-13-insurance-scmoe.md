---
layout: post
title: "SC-MoE: Spatially Coherent Risk Type Discovery for Geographic Motor Pricing"
date: 2026-03-13
author: Burning Cost
categories: [libraries]
tags: [mixture-models, spatial, frequency-severity, python, LRMoE, ADMM, postcode, motor, pricing, insurance-scmoe]
description: "BYM2 and standard GLM spatial smoothing add a continuous geographic effect to a single model. SC-MoE does something fundamentally different: it discovers K latent risk types and uses graph Laplacian regularisation to enforce geographic coherence in which type each postcode belongs to. insurance-scmoe is the Python implementation of the NAAJ 2025 paper."
---

The standard treatment of geography in UK motor pricing is to add a postcode sector effect. Either a crude k-means territory factor, or — if the team is doing it properly — a BYM2 spatial random effect. Either way, the spatial model produces one number per area: a multiplicative loading that shifts the expected claim count up or down.

That framing assumes all geographic variation is a matter of degree. Urban postcodes are riskier than rural postcodes. The London effect is real. Spatial smoothing borrows strength from neighbours so that sparse postcodes get credible estimates. All of that is correct.

But it is not the only thing geography tells you.

Some postcodes are high-frequency, low-severity — congested urban areas where claims are frequent but mostly minor. Others are low-frequency, high-severity — rural roads where incidents happen rarely but the consequences are serious. These are not the same risk type at a different level. They have different frequency-severity dependency structures, different loss distributions, and they should be priced differently in ways a single spatial multiplier cannot capture.

SC-MoE — Spatially Clustered Mixture of Experts — is the model that handles this. The NAAJ 2025 paper by [Fung, Badescu, and colleagues] formalises what that means: discover K latent risk types, assign each policyholder a probability of belonging to each type, and enforce spatial coherence so that nearby postcodes tend to be assigned to the same type. Each type has its own Poisson rate and Gamma severity parameters, and the frequency-severity relationship within each type is modelled jointly.

[`insurance-scmoe`](https://github.com/burning-cost/insurance-scmoe) is the Python implementation.

```bash
pip install insurance-scmoe
```

---

## The model

The core idea is a mixture of experts. Each policyholder is described by a soft assignment across K latent risk classes: π₁(x), π₂(x), …, πK(x), where the πk sum to 1 and depend on the policyholder's covariates x. Within each class k, claims are generated from a joint distribution: Poisson(λₖ) for frequency, Gamma(αₖ, βₖ) for severity. The observable data — policy characteristics, observed claim counts, observed claim amounts — jointly identify the class structure.

The LRMoE (logit-reduced mixture of experts) framework, introduced by Fung, Badescu, and Lin in 2019, is where the actuarial literature developed this. SC-MoE extends it with one additional component: a graph Laplacian spatial penalty on the gating network that enforces geographic coherence in the class assignments.

### Gating network

The class probabilities are produced by a multinomial logistic regression. For class k and policyholder i:

```
log(πk(xi) / πK(xi)) = αk + xi' βk
```

where αk is an intercept and βk is a covariate coefficient vector. All covariate effects enter through the gating. The expert distributions — Poisson rate λk, Gamma shape αk, Gamma rate βk — have no covariates. They are properties of the risk type, not of the individual policyholder.

This is a deliberate design choice with a clear interpretation. The gating asks: given this driver's age, NCD, vehicle type, and postcode, which risk type do they most likely belong to? The experts ask: given that they belong to this risk type, what does their loss distribution look like? The two questions are separated.

### Spatial penalty

The graph Laplacian regularisation operates on the gating intercepts αk — specifically, on their geographic component. Postcodes that are spatially adjacent in the graph are penalised for having different class assignment probabilities. The penalty is:

```
λ · Σk α'k L αk
```

where L is the graph Laplacian of the postcode adjacency graph and λ controls smoothness. At λ = 0, you have standard LRMoE with no spatial structure. As λ grows, adjacent postcodes are pushed toward the same class assignments. At very large λ, the spatial component dominates and all postcodes collapse to a single assignment — complete pooling.

The practical effect is that the K risk types discovered by the model have geographic coherence: class 1 might dominate South London and extend into parts of Surrey, while class 3 is characteristic of rural Scotland, with class 2 occupying an urban-but-not-central intermediate zone. The spatial penalty is not forcing this — it is encouraging it. If the data strongly support a hard geographic boundary, the model will produce one.

---

## Why this is not the same as BYM2

The most important thing to understand about SC-MoE is what it is not doing.

BYM2 adds a spatially smooth random effect to a single GLM. Every area has the same model; the geographic variation is captured by a smooth adjustment to the linear predictor. The BYM2 spatial effect modifies *how much* risk a given area has. It does not change the form of the risk.

SC-MoE discovers that the form of the risk differs by type. Class 1 might have λ₁ = 0.18 claims per year with Gamma shape α₁ = 0.9 — a heavy-tailed severity distribution for relatively frequent, moderately severe claims. Class 3 might have λ₃ = 0.06 with α₃ = 2.8 — infrequent but well-concentrated severity, a very different loss shape. No amount of spatial smoothing on a single GLM's intercept can capture this distinction, because the distinction is in the loss distribution, not just the expected value.

The spatial component also differs mechanically. BYM2 places a prior on the area-level random effects — the prior says nearby areas should have similar *values* of the random effect. SC-MoE penalises the gating network intercepts — the penalty says nearby postcodes should have similar *class assignments*. One is a prior on outcomes; the other is a regularisation on the model's latent structure. They are not interchangeable.

---

## Estimation: MM-ADMM

The model is estimated by an MM-ADMM algorithm — a combination of the Minorisation-Maximisation principle (specifically Böhning's quadratic bound for the multinomial logistic gating) and ADMM (Alternating Direction Method of Multipliers) to handle the penalised gating update.

The obstacle is that the spatial penalty on the gating intercepts breaks the standard closed-form M-step for the logistic regression. ADMM splits the problem into manageable sub-problems: a standard penalised regression update and an unpenalised logistic update, alternating until convergence. The Böhning bound replaces the exact Hessian with a global quadratic majoriser that is invertible without line search.

The implementation is pure numpy and scipy. No PyTorch, no JAX, no compilation step.

```bash
pip install insurance-scmoe
# That is the full dependency tree.
```

---

## Building the spatial graph

Before fitting the model, you need a postcode adjacency graph. The library provides `SpatialGraph` to construct this from the standard sources.

```python
from insurance_scmoe import SpatialGraph

# From a GeoJSON of postcode sector polygons
graph = SpatialGraph.from_geojson(
    "postcode_sectors.geojson",
    area_col="PC_SECTOR",
    connectivity="queen",
)

# From a precomputed adjacency matrix (scipy sparse)
import scipy.sparse as sp
W = sp.load_npz("postcode_adjacency.npz")
graph = SpatialGraph(W=W, areas=sector_codes)

print(f"Areas: {graph.n}")
print(f"Mean neighbours: {graph.mean_degree:.1f}")
print(f"Components: {graph.n_components}")   # must be 1 for the Laplacian
```

If the graph has disconnected components — islands, the Orkneys, overseas UK territories — the library will warn and optionally connect them to their nearest mainland node. The Laplacian requires a connected graph.

---

## Fitting the model

```python
import numpy as np
from insurance_scmoe import SCMoE

# X: (n_policies, p) covariate matrix — standardise continuous covariates
# freq: (n_policies,) integer claim counts
# sev: (n_policies,) claim severity — use 0 (or NaN) for zero-claim policies
# exposure: (n_policies,) earned years
# postcode_idx: (n_policies,) integer index into graph.areas

model = SCMoE(
    n_components=4,       # K risk types — choose via BIC or held-out likelihood
    lam=5.0,              # spatial smoothness penalty
    max_iter=500,
    tol=1e-5,
    random_state=42,
)

result = model.fit(
    X=X,
    freq=freq,
    sev=sev,
    exposure=exposure,
    postcode_idx=postcode_idx,
    graph=graph,
)
```

The `lam` parameter controls spatial smoothness. The right choice depends on the data — a large book with dense postcode coverage can support a smaller lambda (more local variation allowed); a sparse book benefits from heavier smoothing. Cross-validation on held-out log-likelihood, or BIC across a lambda grid, are both implemented.

```python
# Lambda selection by BIC
from insurance_scmoe import lambda_bic_search

lam_opt, bic_curve = lambda_bic_search(
    model_kwargs=dict(n_components=4, max_iter=500, random_state=42),
    lam_grid=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
    X=X, freq=freq, sev=sev, exposure=exposure,
    postcode_idx=postcode_idx, graph=graph,
)
print(f"Optimal lambda: {lam_opt}")
```

---

## Choosing K

K is the number of risk types. There is no oracle answer, and the NAAJ 2025 paper does not pretend otherwise. The practical guidance: start with K = 3 or 4. More than 6 is rarely interpretable in a commercial pricing context — the classes start to blur and the gating probabilities become diffuse.

```python
from insurance_scmoe import component_bic_search

bic_results = component_bic_search(
    K_range=range(2, 8),
    lam=5.0,
    X=X, freq=freq, sev=sev, exposure=exposure,
    postcode_idx=postcode_idx, graph=graph,
    random_state=42,
)
print(bic_results)
#    K      BIC     log_lik    n_params
#    2  142881.2  -71323.4      18
#    3  141203.7  -70459.1      27
#    4  140876.4  -70270.0      36   <-- elbow
#    5  140891.1  -70251.3      45
#    6  140934.0  -70218.7      54
```

BIC typically shows an elbow. The jump from K=3 to K=4 often represents the model distinguishing urban-congested from urban-fast-road from rural-motorway from rural-single-track — a commercially meaningful distinction. Above that, additional classes tend to fractionally repartition existing ones.

---

## Interpreting the output

```python
# Expert parameters: what each risk type looks like
experts = result.expert_params()
print(experts)
#   class  lambda   alpha   beta   mean_sev   cv_sev
#       1   0.183   0.87   0.004    2,175      1.07
#       2   0.094   1.54   0.006    2,567      0.81
#       3   0.061   2.91   0.008    3,638      0.59
#       4   0.029   1.19   0.006    1,983      0.92

# Class membership: soft assignment per policy
probs = result.class_probs()    # (n_policies, K) array
assigned = probs.argmax(axis=1) # hard assignment for inspection

# Geographic distribution of classes
geo_summary = result.geographic_summary(graph=graph)
print(geo_summary.head(10))
# postcode | dominant_class | class_1_prob | class_2_prob | ...

# Pure premium per policy, integrating over class uncertainty
pp = result.pure_premium(X=X, postcode_idx=postcode_idx, exposure=exposure)
```

The expert parameters are the model's read of the K risk types. In the example above: class 1 is high-frequency, high-CV severity — urban congested, many small claims, heavy tail. Class 3 is low-frequency, low-CV severity — rural, infrequent incidents, well-behaved loss amounts. Classes 2 and 4 are intermediate types.

The pure premium calculation accounts for the frequency-severity dependency within each class. This matters. In a class with Poisson(0.18) frequency and Gamma(0.87, ...) severity, the coefficient of variation of severity is 1.07 — the distribution is heavy-tailed and the covariance between frequency and severity matters for the expected aggregate loss. The standard two-GLM formula, which assumes independence and just multiplies E[N] × E[S], understates the expected loss in classes with high-CV severity. SC-MoE gets this right automatically because frequency and severity are modelled jointly within each class.

---

## Simulation

The library includes a simulator for testing and validation.

```python
from insurance_scmoe import simulate_scmoe

# Generate synthetic data from a known SC-MoE structure
data = simulate_scmoe(
    graph=graph,
    n_policies_per_area=50,
    n_components=4,
    true_lam=5.0,
    expert_params=[
        dict(lambda_=0.18, alpha=0.87, beta=0.004),
        dict(lambda_=0.09, alpha=1.54, beta=0.006),
        dict(lambda_=0.06, alpha=2.91, beta=0.008),
        dict(lambda_=0.03, alpha=1.19, beta=0.006),
    ],
    random_state=42,
)

# data.X, data.freq, data.sev, data.exposure, data.postcode_idx, data.true_classes
```

The simulator draws class assignments from the spatial prior, then samples frequency and severity from the class-specific distributions. Use it to validate recovery of parameters before fitting to real data, and to test how sensitive the class structure is to the choice of K.

---

## What it does not do

SC-MoE is a mixture model, not a GLM. There are no individual-level multiplicative relativities for vehicle make, NCD band, or driver age that a pricing committee can read off a table. The covariates enter through the gating — they shift class probabilities — and the class structure carries the risk. If you need the output to be a standard GLM relativity table, SC-MoE is the wrong model.

It is also not a replacement for territory banding in a rating engine. The class membership probabilities are policy-level outputs of the model. They are not, in themselves, rating factors that slot into Emblem or Radar. The practical path is to use the geographic summary — the dominant class assignment per postcode sector — as one input to a territory factor derivation, alongside other area-level information.

The model also requires that you have both frequency and severity data. If you are working with aggregate loss ratios, or a line where severity is not observed separately from frequency (certain commercial liability structures, for example), SC-MoE does not apply.

---

## In relation to what we have already built

We have now built several libraries that touch on related problems, and the distinctions are worth stating clearly.

[`insurance-spatial`](/2026/03/09/spatial-territory-ratemaking-bym2/) uses BYM2 to add a smooth spatial random effect to a single GLM. It answers: how much higher or lower is claim frequency in this postcode sector, after controlling for vehicle and driver characteristics? SC-MoE answers: which risk type does this postcode sector belong to, and what are the joint frequency-severity parameters of that type? These are different questions. A team doing BYM2 territory factors is not making SC-MoE redundant.

[`insurance-nested-glm`](/2026/03/09/nested-glms-with-neural-network-embeddings-for-insurance/) handles high-cardinality categorical variables — vehicle make, postcode sector — by learning embeddings and then fitting a standard GLM. The output is a GLM relativity table. SC-MoE produces class membership probabilities and class-specific distributional parameters. They are solving different problems.

[`insurance-glm-cluster`](/2026/03/13/insurance-glm-cluster/) collapses factor levels within an existing GLM framework. It is not a mixture model and does not discover latent types.

[`insurance-frequency-severity`](/2026/03/12/insurance-frequency-severity/) captures frequency-severity dependency via a global Sarmanov copula: one copula structure, one correlation parameter, across the whole book. SC-MoE captures it locally — within each of K classes — which allows the dependency structure to vary by risk type. Both are improvements on the standard two-GLM approach that assumes independence. They are not direct substitutes: Sarmanov is faster, simpler, and more auditable; SC-MoE is more flexible but substantially harder to explain to a pricing committee.

---

## On the paper

The SC-MoE model is from: "Spatially Clustered Mixture of Experts Model for Dependent Frequency and Severity of Insurance Claims," published in the North American Actuarial Journal in 2025 (DOI: 10.1080/10920277.2025.2567283). The paper builds on the LRMoE framework of Fung, Badescu, and Lin (2019a, 2019b) and the graph Laplacian regularisation literature in spatial statistics. The MM-ADMM estimation scheme is the paper's main algorithmic contribution — the Böhning quadratic bound is what makes the spatial penalty tractable without requiring a general-purpose optimiser.

The paper is readable. Read Section 3 (the model) before using the library. Section 4 (estimation) is worth understanding before tuning the algorithm. Section 5 (simulation study) gives you a benchmark for what recovery looks like with N=5,000 policies and K=3 classes.

---

**[insurance-scmoe on GitHub](https://github.com/burning-cost/insurance-scmoe)** — MIT-licensed, PyPI.

**See also:**
- [Getting Spatial Territory Factors Into Production](/2026/03/09/spatial-territory-ratemaking-bym2/) — BYM2 for postcode territory factors, the spatial alternative
- [Nested GLMs with Neural Network Embeddings for Insurance](/2026/03/09/nested-glms-with-neural-network-embeddings-for-insurance/) — handling high-cardinality geographic variables in a GLM
- [500 Vehicle Makes, One Afternoon, Zero Reproducibility](/2026/03/13/insurance-glm-cluster/) — factor level collapsing within a GLM framework
- [Two GLMs, One Unjustified Assumption](/2026/03/12/insurance-frequency-severity/) — frequency-severity dependence via Sarmanov copula
