---
layout: post
title: "Pairwise Interaction Networks: The Model That Beats GBMs and Prints as a Relativities Table"
date: 2026-03-13
categories: [libraries, pricing, interpretability]
tags: [insurance-pin, neural-ga2m, interpretability, fca-consumer-duty, pairwise-interactions, pytorch, french-mtpl]
description: "Pairwise Interaction Networks produce exact tabulatable 2D rating factor surfaces, not SHAP approximations. Beats GBMs on French MTPL benchmark. Python."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-pin` has been consolidated into [`insurance-gam`](/insurance-gam/). Install with `pip install insurance-gam` — all functionality described here is available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-gam)
</div>


Post #102 | insurance-pin v0.1.0

---

Here is the standard story in UK motor pricing. The team fits a GBM for frequency. It outperforms the production GLM by a meaningful margin -- say, 0.8% on held-out Poisson deviance. The actuarial director asks: "Can we take this to committee?" Someone produces SHAP plots. The committee members, who know what a relativities table looks like and have opinions about BonusMalus, look at a scatter of dots on a waterfall chart and say: "What exactly is the model doing with van drivers over 70?" Nobody can answer with precision. The GBM stays off the tariff.

This dynamic is not unique to GBMs. CANNs -- neural networks used as corrections on top of a GLM -- have the same problem. The correction is a black box. GBM residual models used as interaction detectors: same problem. SHAP, however accurate on average, is an approximation. It is not the model.

Richman, Scognamiglio, and Wuthrich (arXiv:2508.15678, August 2025) have a different answer. PIN -- Pairwise Interaction Networks -- decomposes prediction into an exact sum of named terms. Main effects: one function per feature. Pairwise interactions: one function per feature pair. The decomposition is not post-hoc. It is the model. Every pairwise interaction surface can be printed as a 2D rating factor table and handed to a committee member.

It also produces the best published out-of-sample Poisson deviance on the French MTPL benchmark (freMTPL2freq, n=678,007). Better than GLMs, GBMs, CANNs, and the Credibility Transformer. With 4,147 parameters.

We have built [`insurance-pin`](https://github.com/burning-cost/insurance-gam).

---

## What PIN actually is

The prediction equation is:

```
mu_i = exp( b + sum_j f_j(x_j) + sum_{j<k} f_{jk}(x_j, x_k) )
```

This is GA2M -- the Generalised Additive Model with pairwise interactions that Lou, Caruana, Gehrke, and Hooker introduced at KDD 2013. The main effects `f_j(x_j)` behave exactly like GAM shape functions: one curve per feature. The pairwise terms `f_{jk}(x_j, x_k)` are two-dimensional surfaces: the BonusMalus x VehBrand interaction, as a table.

PIN's novelty is not the equation. It is the parameterisation.

In the original GA2M paper, each term is a boosted lookup table, fitted stage-wise. In EBM (InterpretML's production implementation), same approach: cyclic gradient boosting. PIN replaces every term with a neural network output, and all terms share a single network backbone distinguished by learned interaction tokens.

The architecture for any pair (j, k):

```
h_{j,k}(x) = sigma_hard( f_theta(phi_j(x_j), phi_k(x_k), e_{j,k}) )
```

- `phi_j` is a feature-specific embedding: a small two-layer network (continuous features) or a learnable lookup table (categorical features).
- `e_{j,k}` is a learned interaction token -- a vector in R^10, one per pair, that tells the shared network "this is the BonusMalus x VehBrand surface, not the DrivAge x Density surface".
- `f_theta` is a three-layer feedforward network, shared across all 45 pairs.
- `sigma_hard` is a centred hard sigmoid: `clamp((1 + x)/2, 0, 1)`. The output lives in [0, 1]. Multiple units with scalar weights produce piecewise-constant approximations of each shape function -- analogous to tree leaf values, which is where the "tree-like" name comes from.

The scalar weights `w_{j,k}` on each term are learnable, real-valued, and unconstrained. They determine how much each pairwise interaction contributes to the final prediction, and can be zero or negative.

For q=9 features, there are 9 main effects and 36 pairwise interactions -- 45 terms total, each passing through the same network with a different token. This is why the parameter count is 4,147 rather than the tens of thousands you would expect from 45 separate networks.

One implementation note that will save you an afternoon: `torch.nn.Hardsigmoid` uses `clamp((x+3)/6, 0, 1)` -- a different formula with different breakpoints. Do not use it. The correct implementation is:

```python
def centered_hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((1.0 + x) / 2.0, min=0.0, max=1.0)
```

The paper is silent on this distinction, and the mismatch will produce subtly wrong outputs without raising an error.

---

## What you get at the end

After training, you have exact expressions for every pairwise surface. Not SHAP approximations. The surface `f_{jk}(x_j, x_k)` is computed by passing a grid of (x_j, x_k) values through the model with all other features held at baseline. The result is a two-dimensional array that tabulates directly as a multiplicative rating factor.

Features are specified as a plain dict mapping feature name to either `"continuous"` or the number of category levels (an integer). Categorical features expect integer codes starting at zero.

```python
import polars as pl
from insurance_gam import PINModel, PINDiagnostics

features = {
    "DrivAge": "continuous",
    "BonusMalus": "continuous",
    "VehPower": "continuous",
    "Density": "continuous",
    "VehBrand": 11,   # 11 category levels, integer-coded 0..10
    "Region": 22,     # 22 regions
}

model = PINModel(
    features=features,
    embedding_dim=10,
    hidden_dim=20,
    token_dim=10,
    shared_dims=(30, 20),
    loss="poisson",
    lr=0.001,
    max_epochs=500,
    patience=20,
)

# X_train is a Polars DataFrame or dict of arrays; y_train is a numpy array
# of observed frequency (claims / exposure); exposure is years at risk
model.fit(X_train, y_train, exposure=exposure_train, verbose=True)

# Predictions
freq_hat = model.predict(X_val)                        # numpy array, shape (n,)
freq_hat_with_exp = model.predict(X_val, exposure=exposure_val)
```

The interaction surfaces live in a dict keyed by `(feature_j, feature_k)` tuples. Each value contains the grid coordinates and the surface values on the linear predictor scale:

```python
diag = PINDiagnostics(model)

# Plot the BonusMalus x VehBrand interaction surface
fig, ax = diag.plot_surface("BonusMalus", "VehBrand", X_background=X_train, n_grid=50)
fig.savefig("bonus_vehbrand_surface.png", dpi=150)

# Or work with the raw arrays directly
surfaces = model.interaction_surfaces(X_train, n_grid=50, pairs=[("BonusMalus", "VehBrand")])
surf = surfaces[("BonusMalus", "VehBrand")]
# surf["grid_j"]: BonusMalus values, shape (50,)
# surf["grid_k"]: VehBrand codes, shape (11,)
# surf["surface"]: w*h values, shape (50, 11) -- linear predictor scale
# exp(surf["surface"]) gives multiplicative relativities

import numpy as np
relativities = np.exp(surf["surface"])   # 50 x 11 table of multiplicative factors
```

The `exp()` step converts the linear predictor scale to multiplicative relativities. This is a table that can be handed to a committee member and loaded into a rating engine, row by row, exactly as GLM relativities are. Not an approximation. The model's actual output.

To see which pairs are worth modelling at all, rank by importance before committing to the full 36:

```python
# Importance: range of w_{jk} * h_{jk}(x) on training data
# Large weight, flat surface = low importance. This is the right metric.
fig, ax, importance_dict = diag.weighted_importance(X_train, top_n=10)

# Fit an ensemble of 10 models for variance reduction (recommended for production)
from insurance_gam import PINEnsemble

ensemble = PINEnsemble(
    n_models=10,
    features=features,
    loss="poisson",
    embedding_dim=10,
    hidden_dim=20,
    token_dim=10,
    shared_dims=(30, 20),
)
ensemble.fit(X_train, y_train, exposure=exposure_train)
freq_ensemble = ensemble.predict(X_val)
uncertainty = ensemble.predict_std(X_val)   # epistemic uncertainty across models
```

A model fitted with the top five pairs identified by `weighted_importance()` is considerably easier to present than one with 36. Which five actually matter is an empirical question, answered by the data rather than by prior judgement.

---

## Where it sits against what you already use

| Model | Exact interaction surfaces | FCA-auditable decomposition | freMTPL2freq Poisson deviance | Parameters |
|---|---|---|---|---|
| GLM | No | Yes (main effects only) | ~23.90 | ~20 |
| EBM (InterpretML) | Yes (lookup tables) | Yes | ~23.75 | Thousands of split points |
| CANN / NID | No | No (correction is a black box) | ~23.68 | Varies |
| PIN (Richman et al. 2025) | Yes (neural surfaces) | Yes | **23.63** | 4,147 |

Deviance on the 10^-2 scale; lower is better. PIN's 23.63 is the best published figure for any interpretable model on this benchmark as of August 2025.

The CANN comparison is the one that matters in practice. CANNs achieve similar predictive performance but the neural correction layer is unauditable by design. PIN achieves better performance and full decomposition by design. These are not competing objectives that have been traded off; PIN is strictly better on both.

Two honest caveats. The paper does not benchmark against a tuned CatBoost or LightGBM on the same dataset. Our expectation is that a well-tuned GBM would sit around 23.65-23.70 -- close to PIN but without the interpretable structure. We will run that comparison when the library is built. And PIN has no native monotonicity constraints. "Claims frequency must not decrease with BonusMalus level" is a requirement in some UK tariff submissions. ANAM has this natively; PIN does not in v1.

---

## When to use PIN

Use it when you need GBM-level accuracy and GLM-level auditability simultaneously, and when the specific question from your actuarial committee is "what exactly is the interaction between X and Y doing?"

The FCA Consumer Duty requirement for explainability of pricing factors is the clearest regulatory driver here. A SHAP waterfall chart is defensible under Consumer Duty. An exact additive decomposition with named, tabulatable pairwise surfaces is stronger. The difference matters when a regulator asks for a precise account of why a specific customer received a specific price. "SHAP approximation" is a weaker answer than "here is the exact contribution of each named factor."

Do not use PIN as a first model. Start with a GLM to understand your main effects. Run an EBM to see whether shape functions improve on the GLM's linearity assumptions. Add PIN when you want to identify which pairwise interactions are genuinely driving deviance improvement, and when you need those interactions in a form that a committee can approve and a rating system can consume.

The library is PyTorch throughout. The authors released a ZIP archive at github.com/wueth/Tree-Like-PIN that appears to be Keras/TensorFlow -- consistent with Wuthrich group conventions. We have re-implemented from the paper equations in PyTorch, tested against the reference parameter count of 4,147 as a regression anchor.

The repository is at [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam).

---

## References

Richman, R., Scognamiglio, S., and Wuthrich, M.V. "Tree-like Pairwise Interaction Networks." arXiv:2508.15678, August 2025.

Lou, Y., Caruana, R., Gehrke, J., and Hooker, G. "Accurate Intelligible Models with Pairwise Interactions." KDD 2013.

- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Explainable Boosting Machines for Insurance Tariff Construction](/2026/03/09/explainable-boosting-machines-for-insurance-pricing/)
- [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/)
