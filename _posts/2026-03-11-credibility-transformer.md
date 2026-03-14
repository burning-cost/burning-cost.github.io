---
layout: post
title: "The Attention Head That Is Also a Credibility Weight"
date: 2026-03-11
categories: [libraries, pricing, credibility]
tags: [credibility-transformer, Bühlmann-Straub, attention, transformer, CLS-token, Wüthrich, ICL, zero-shot, PyTorch, MTPL, frequency, insurance-credibility-transformer, python]
description: "Credibility Transformer attention is mathematically identical to Buhlmann-Straub credibility. PyTorch, 1,746 parameters, beats CAFTT (27K). 85 tests."
---

There is a common pattern in machine learning papers applied to insurance. The authors train a neural network, show it beats a GLM on a benchmark, and then add a section titled "Interpretability" where they note that some internal representation "resembles" a credibility weight, or "can be interpreted as" a posterior probability, or "is analogous to" a Bühlmann parameter. The resemblance is gesturing at something real but the connection is always approximate, always informal, always one level of abstraction away from actually being true.

The Credibility Transformer (Richman, Scognamiglio, Wüthrich, 2024; arXiv:2409.16653, published European Actuarial Journal 2025) does something different. The CLS token's self-attention weight is the Bühlmann-Straub credibility weight. Not resembles. Not is analogous to. Is. The mathematical identity is proven in the paper, not asserted.

[`insurance-credibility-transformer`](https://github.com/burning-cost/insurance-credibility-transformer) implements the base CT and its ICL extension in PyTorch. 85 tests, 2,372 lines of source, MIT-licensed, on PyPI.

```bash
pip install insurance-credibility-transformer
```

---

## The architecture

The CT is built on the FT-Transformer (Gorishniy et al., 2021) adapted for tabular insurance data. The key structural choice is the CLS token.

Standard attention in an FT-Transformer takes a set of feature embeddings and mixes them. The CT appends a learned CLS token to the feature sequence, so the attention mechanism operates on T+1 tokens (T features plus the CLS token). After the Transformer layers, the CLS token is decoded to produce the frequency prediction.

The input tokenisation is standard: categorical covariates go through entity embeddings into R^b, continuous covariates go through a two-layer FNN into R^b, all features are embedded to the same dimension b, and positional encodings are concatenated. For the French MTPL benchmark: 4 categorical covariates (Area, VehBrand, VehGas, Region) and 5 continuous (VehPower, VehAge, DrivAge, BonusMalus, Density), embedded to b=5 for the base model.

Two versions of the CLS token are maintained:

- **c_prior**: the CLS token passed through the FNN blocks without attention. This sees no individual covariate information. It learns to encode the portfolio average.
- **c_trans**: the CLS token after full self-attention. This has attended to all T feature embeddings and encodes individual risk information.

During training, a Bernoulli sample Z ~ Bernoulli(alpha) selects which token to forward. When Z=1, the model trains on individual covariate information. When Z=0, it trains to encode the portfolio mean. With alpha=0.90 (the paper's grid-searched value), 90% of gradient steps use the individual token and 10% force the prior to track the portfolio average. This acts as a credibility-regularised dropout.

---

## Why the self-attention weight is the credibility weight

Let P = a_{T+1, T+1}: the attention weight the CLS token assigns to itself in the self-attention matrix. The CLS token's attended output is:

```
v_trans = P * v_CLS + (1-P) * v_covariate

where v_covariate = sum over t=1..T of (a_{T+1,t} / (1-P)) * v_t
```

The CLS value vector v_CLS encodes the portfolio prior (because the CLS token was initialised as a learned prior and trained via the Bernoulli mechanism to represent the portfolio mean). The covariate value vectors v_1..v_T encode the individual risk information.

The Bühlmann-Straub credibility estimate is:

```
mu_hat = Z * Y_bar_individual + (1-Z) * mu_prior
```

where Z is the credibility weight, Y_bar_individual is the individual mean, and mu_prior is the portfolio prior.

P and Z play the identical role. The CLS self-attention weight is the credibility weight, learned end-to-end from data. It is not fixed — it varies by risk, by exposure, by the covariate configuration of each observation. Bühlmann-Straub gives you a single Z per group derived from the between/within variance ratio. The CT gives you a P per policy, learned from the attention mechanism, that encodes a richer version of the same idea.

The decoder is a two-layer FNN with exponential activation:

```
mu_CT(x) = exp(FNN(c_cred(x)))
```

Loss is exposure-weighted Poisson deviance — the correct strictly consistent scoring rule for frequency estimation.

---

## Numbers

French MTPL: 610,206 learning policies, 67,801 test policies, 9 covariates. Out-of-sample Poisson deviance (x 10^-2, lower is better):

```
Null model:          25.445
GLM:                 24.102
FNN ensemble:        23.783
CT nadam ensemble:   23.711  (1,746 parameters)
CAFTT ensemble:      23.726  (27,133 parameters)
Deep CT ensemble:    23.577  (320K parameters, GPU required)
```

The base CT beats CAFTT (the Constrained Attention Feature-based Tabular Transformer from Brauer, 2024) with 1,746 parameters versus CAFTT's 27,133. That is not a different point on the same efficiency curve — it is a different regime entirely. The CAFTT result is from a model that knows nothing about credibility theory. The CT result comes from baking that theory directly into the architecture.

The gap over the FNN ensemble (23.711 vs 23.783) is modest in absolute terms. That is honest. The CT is not claiming to revolutionise motor pricing. On a 610K-row MTPL benchmark, it improves Poisson deviance by about 0.3% over a standard neural ensemble. What it provides that the FNN cannot is the credibility interpretation — which matters for governance, for communication with actuarial committees, and for knowing what the model is actually doing.

---

## ICL extension: zero-shot pricing for unseen risks

The In-Context Learning extension (Padayachy, Richman, Scognamiglio, Wüthrich, arXiv:2509.08122v2, January 2026) changes the inference protocol. Instead of evaluating each risk in isolation, ICL-CT augments each prediction with a context batch of similar policies whose outcomes are known.

This is not fine-tuning or retraining. At inference time, for each target risk:

1. Retrieve K=64 nearest neighbours from the training set using cosine similarity in the CLS token embedding space (FAISS approximate nearest neighbour search)
2. Decorate the context instance CLS tokens with observed outcome information via a learned FNN
3. Pass target and context tokens through a causal attention layer — target tokens can attend to context tokens but not to each other
4. The causal attention weights are themselves Bühlmann credibility weights: context instances with higher exposure receive more weight via v_j / (v_j + kappa), where kappa is a learned credibility coefficient

The outcome token decoration formula is:

```
c_decor(x_i) = c_cred(x_i) + (v_i / (v_i + kappa)) * FNN(Y_i)
```

where v_i is exposure and Y_i is the observed claim count. Higher exposure makes the observed outcome more informative — exactly the Bühlmann credibility weighting applied now to context instances at inference time.

The practical consequence: ICL-CT can price risks with covariate combinations not seen during training. A new vehicle model launching in March that has no training history. A new postcode sector in a geographic expansion. A commercial class that was not in the training data. The model retrieves similar risks from the training set, uses their outcomes as in-context experience, and produces a price without retraining.

ICL results on French MTPL (5 runs, out-of-sample Poisson deviance x 10^-2):

```
Base CT phase 1:           23.743
ICL 2-layer phase 2:       23.725
ICL fine-tuned phase 3:    23.710
Ensemble ICL 2-layer:      23.679
```

The ICL improvement over the base CT ensemble is about 0.03 Poisson deviance units — consistent but modest. The stronger argument for ICL-CT is the zero-shot capability, not the benchmark number.

---

## Three training phases

**Phase 1**: Train base CT from scratch. AdamW (LR=1e-3, weight decay=1e-2, beta2=0.95), 100 epochs, batch size 1024, early stopping with patience 20.

**Phase 2**: Freeze the decoder from Phase 1. Train the outcome token decorator and ICL Transformer layer. AdamW (LR=3e-4), 50 epochs. The frozen decoder preserves calibration from Phase 1 while the ICL layer learns to use context information.

**Phase 3** (optional): Full fine-tuning. AdamW (LR=3e-5), 20 epochs. Phase 3 consistently improves on Phase 2 but adds training time. Whether it is worth it depends on your context.

---

## Quick start

```python
from insurance_credibility_transformer import CredibilityTransformer
import pandas as pd

# df: policy-level DataFrame
# Claims column: integer claim counts
# Exposure column: fraction of year at risk
# Categorical columns: will be entity-embedded
# Continuous columns: standardised internally

ct = CredibilityTransformer(
    embedding_dim=5,
    alpha=0.90,
    n_runs=5,
    loss="poisson_deviance",
)

ct.fit(
    df_train,
    claims_col="claim_count",
    exposure_col="exposure",
    categorical_cols=["area", "vehicle_brand", "fuel_type", "region"],
    continuous_cols=["vehicle_power", "vehicle_age", "driver_age", "bonus_malus", "density"],
)

# Ensemble mean of n_runs models
mu_hat = ct.predict(df_test)

# Inspect attention-as-credibility weights
# P_i: credibility weight for each test observation
P = ct.credibility_weights(df_test)
print(f"Mean P: {P.mean():.3f}  (range {P.min():.3f}–{P.max():.3f})")
```

The `credibility_weights()` method returns P for each observation — the CLS self-attention weight that is the Bühlmann-Straub credibility weight. This is not a post-hoc explanation. It is the actual quantity the model used to form its prediction, read directly from the attention matrix.

For ICL-CT:

```python
from insurance_credibility_transformer import ICLCredibilityTransformer

icl_ct = ICLCredibilityTransformer(
    base_ct=ct,  # pre-trained phase 1 model
    context_k=64,
    context_pool_size=1000,
    kappa_init=1.0,
)

# Phase 2 training
icl_ct.fit_icl(df_train, claims_col="claim_count", exposure_col="exposure")

# Inference: retrieves context automatically
mu_icl = icl_ct.predict(df_test, reference_pool=df_train)

# Zero-shot: works even if df_new contains vehicle models not in df_train
mu_new = icl_ct.predict(df_new, reference_pool=df_train)
```

---

## Where to use this and where not to

**Base CT** is the right choice for UK personal lines motor with a large dataset — call it 100K+ policies. It trains in minutes on CPU (the base model has 1,746 parameters). The credibility interpretation gives you something to show a pricing committee that a GBM cannot: the model's prior estimate, the model's individual estimate, and the weight assigned to each. You can extract P per policy and show that thin segments (new vehicles, rural postcodes with 50 policies) receive lower P values — meaning the model relies more on the portfolio prior — which is exactly what credibility theory says should happen.

**ICL-CT** is worth adding when you have recurring new-to-market risks. New vehicle models launch every quarter. Geographic expansion into new postcodes. Commercial lines where a new scheme has no history but similar existing schemes do. The three-phase training is more involved and requires building and maintaining a FAISS index on the training set, but the zero-shot capability is the only approach in the academic literature that handles unseen categorical levels without retraining.

**What it is not**: a replacement for Bühlmann-Straub credibility on aggregated scheme data. Bühlmann-Straub operates on group × year panels — broker adjustments, scheme differentials, fleet pricing. It is fully auditable and requires no GPU. The CT operates at individual policy level on raw covariates. These are different problems. Our [`credibility`](https://github.com/burning-cost/insurance-credibility) library handles the aggregated use case.

**Deep CT** (320K parameters, multi-head attention, SwiGLU gating, differentiable PLE for continuous features) achieves Poisson deviance of 23.577 — a meaningful improvement over the base. It is marked as experimental in v0.1. It requires a GPU (about 7 minutes per run on an L4) and the performance gain over the base is only visible at larger dataset sizes. We are building the infrastructure around it first.

---

## Governance considerations

The CLS token self-attention weight P provides a form of interpretability that is unusual for neural network pricing models: it is not a post-hoc SHAP value or a saliency map added after training. It is a quantity the model was explicitly designed to produce and directly uses in forming its prediction.

For FCA purposes, the attention mechanism is still harder to explain than GLM coefficients or GBM SHAP values. P tells you how much individual versus portfolio information the model used, but not which individual features drove the individual estimate. That is a gap. The feature-level attention weights (the a_{T+1, t} values, t=1..T) give you a partial answer — they show which features the CLS token attended to — but interpreting attention weights as feature importances has well-known problems. We do not hide this.

What the CT gives you, uniquely, is a principled prior/posterior decomposition at policy level. That is not available from a GBM, and it is what Bühlmann-Straub gives you at group level. Whether that is sufficient for your regulatory context is a judgment call, not a property of the model.

---

**[insurance-credibility-transformer on GitHub](https://github.com/burning-cost/insurance-credibility-transformer)** — 85 tests, MIT-licensed, PyPI. Library #45.

---

**Related reading:**
- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/) — the frequentist baseline: moment-based credibility without MCMC or neural networks
- [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/) — full posterior inference across hierarchy levels; the Bayesian alternative to the attention-based approach
- [Individual Experience Rating Beyond NCD](/2026/03/13/insurance-experience/) — individual-level Bayesian a posteriori rating; complements the Credibility Transformer's group-level approach with policy-level posterior updating
