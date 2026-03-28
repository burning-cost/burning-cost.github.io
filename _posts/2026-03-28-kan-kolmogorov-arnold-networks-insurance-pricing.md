---
layout: post
title: "KAN: What If Your Neural Network Learned Its Own Link Functions?"
date: 2026-03-28
categories: [pricing, neural-networks, interpretability]
tags: [kan, kolmogorov-arnold, neural-network, splines, gam, glm, link-function, monotonicity, symbolic-regression, poisson, insurance-pricing, mortality, python, pykan, monokan, interpretability, governance]
description: "Kolmogorov-Arnold Networks replace fixed activations with learnable splines on edges, letting the model discover its own functional forms. Here is what that means for insurance pricing, and what it does not mean."
seo_title: "KAN (Kolmogorov-Arnold Networks) for Insurance Pricing: Architecture, MonoKAN Monotonicity, and Symbolic Regression"
---

Every GLM you have ever built contains a human decision that is not in the data: the link function. When you write `family=Poisson(link='log')`, you are asserting that the relationship between your rating factors and expected claim frequency is log-linear. Not log-quadratic. Not something more interesting for young drivers at high vehicle groups. Log-linear, throughout.

That choice is defensible — it usually works, it produces stable coefficients, and regulators understand it. But it is a constraint, not a discovery. Kolmogorov-Arnold Networks (KANs) are the first architecture we have seen that makes a coherent argument for relaxing it in a way that is still interpretable at the end.

This is not a "GBMs are uninterpretable and KANs will replace them" piece. KANs do not consistently beat gradient boosting on tabular data, and they train more slowly. The honest case for them is narrower and more interesting: for problems where you need to show regulators what the model is actually doing — not a SHAP approximation, but the actual learned functional forms — KANs are worth understanding.

---

## The problem with fixed link functions

A standard Poisson GLM for claim frequency looks like:

```
log(E[claims_i]) = log(exposure_i) + β₀ + β₁·age_i + β₂·ncd_i + ...
```

The `log()` on the left side is the link function. It ensures predicted frequencies are positive and multiplicatively separable by factor. These are good properties. But they come at a cost: you have pre-committed to the functional form before seeing the data.

GAMs (and our [`insurance-gam`](https://github.com/burning-cost/insurance-gam) library) partially solve this. They replace the linear terms with smooth functions:

```
log(E[claims_i]) = log(exposure_i) + f₁(age_i) + f₂(ncd_i) + ...
```

Each `f_j` is a spline — typically penalised B-splines fitted by marginalising over a prior on smoothness. This lets age have a non-linear effect without requiring manual bucketing. The shape functions are readable: you can plot `f₁(age)` and hand it to a pricing committee.

But GAMs still require you to choose the link function. And the additive structure — one term per feature — is imposed, not learned. If the actual relationship is `f(age, ncd_interaction)`, GAM will miss it unless you explicitly include a tensor product term.

KAN does something different.

---

## The architecture

The key paper is Liu et al. (2024), "KAN: Kolmogorov-Arnold Networks" (arXiv:2404.19756), which appeared at ICLR 2025. The name comes from the Kolmogorov-Arnold representation theorem, a 1957 result that says any continuous multivariate function `f(x₁, ..., xₙ)` can be written as a finite composition of continuous univariate functions and addition:

```
f(x₁, ..., xₙ) = Σ_q Φ_q( Σ_p φ_{q,p}(x_p) )
```

This is not a practical implementation recipe — the inner functions `φ_{q,p}` in the original theorem are continuous but generally not smooth. What Liu et al. did was ask: what if we parameterise those univariate functions as B-splines and learn them from data?

The result is an architecture where the "weights" are not scalars — they are spline curves. A conventional MLP multiplies each activation by a learned scalar and passes the result through a fixed function (ReLU, sigmoid, tanh). A KAN replaces each scalar weight with a learnable one-dimensional spline. The network applies these edge-wise splines to the inputs, sums across incoming edges, then passes to the next layer.

What this buys you:

**Visualisability.** Every edge in a trained KAN corresponds to a learned function you can plot. For a one-hidden-layer KAN with 4 inputs and 1 output, you have at most `4 × hidden_size + hidden_size × 1` learnable curves, each printable as a graph. There is no need to invoke SHAP or ICE plots — the learned functions are the model.

**Symbolic regression.** Once a KAN is trained, the `suggest_symbolic` method in the `pykan` library attempts to match each learned spline to a closed-form expression from a library of candidates (polynomials, logarithms, exponentials, trig functions). If the data support it, a KAN trained on claim frequency might discover that one edge approximates `log(1 + age/20)`. You can then snapshot that expression and put it in a governance document. This is categorically different from SHAP: SHAP approximates feature importance post-hoc; symbolic regression reads off the model's actual learned functional form.

**Compositional generalisation.** Because KAN learns the entire functional structure compositionally — not just the shape of each feature's effect, but also the structure of how features combine — it can represent interactions that additive models miss.

---

## What the actuarial literature says

The first actuarial application is Xu, Badescu, and Pesenti (2026), "What KAN mortality say: smooth and interpretable mortality modeling using Kolmogorov-Arnold networks," published in the *ASTIN Bulletin* 56(1), pp. 32–59 (DOI: [10.1017/asb.2025.10079](https://doi.org/10.1017/asb.2025.10079)).

Their setting is mortality modelling — life, not non-life — but the transferable findings are substantial. They test three variants: KANN[2,1] (a pure two-hidden-layer KAN), KANNLC (a KAN combined with a Lee-Carter factor structure), and KANNAPC (KAN combined with an age-period-cohort structure). They initialise each variant from pre-fitted traditional actuarial models, using the GLM log-relativities to set the initial spline shapes before gradient descent begins.

Their results across 34 populations: KANN variants outperform the CANN (Combined Actuarial Neural Network) baseline — MLP-based skip connection on top of a classical model — in most cases. The mortality curves produced by KANN are smooth as a direct consequence of spline regularisation, not post-processing. For governance purposes, this matters: the smoothness is a mathematical property of the model, not an artefact of how you plotted the output.

The CANN comparison is relevant for UK non-life pricing because CANN is the architecture already discussed in industry for GLM+neural-network hybrid approaches. KANN is in the same family — initialise from a classical actuarial model, fine-tune with gradient descent — but with an architecture that produces directly readable learned functions rather than an opaque residual correction.

The paper covers mortality. No published actuarial study has yet applied KAN to non-life Poisson frequency or Gamma severity on a standard dataset like French MTPL. That gap exists.

---

## Monotonicity for regulatory purposes

Xu et al. train on mortality, where monotonicity requirements are less pressing. For UK motor and property pricing, they are not optional.

The FCA's consumer duty pricing rules require firms to be able to explain and justify pricing factors. In practice, "vehicle age increases risk" needs to be monotone: if your model prices a 12-year-old car lower risk than a 10-year-old car of the same type, you will need to explain why. The same applies to NCD discount trajectories. Regulators and pricing committees are not sympathetic to "the model found a local optimum."

Standard KANs provide no monotonicity guarantee. Polo-Molina, Alfaya, and Portela (2024), "MonoKAN: Certified Monotonic Kolmogorov-Arnold Network" (arXiv:2409.11078), address this directly. They replace B-splines with cubic Hermite splines parameterised so that the derivative is always non-negative (or non-positive) by construction. The monotonicity is certified: it is a mathematical property of the parameterisation, not a penalty term that could be violated by a sufficiently large gradient step.

This is the key distinction from how [`insurance-gam`](https://github.com/burning-cost/insurance-gam) handles it. The EBM wrapper in `insurance-gam` accepts a `monotone_constraints` dict, which passes through to interpretML's boosting algorithm as a constraint on which tree splits are permitted. That is an approximately-monotone solution — it works well in practice and passes audit, but it is not *certified* in the MonoKAN sense. MonoKAN's guarantee is stronger and more defensible under a formal model governance review.

---

## How KAN fits into our existing model landscape

We get this question frequently, so we will be direct.

**KAN vs GAM/EBM (insurance-gam):** GAMs (including EBMs) are additive — prediction is a sum of per-feature terms. KAN is compositional — the learned functions compose in layers, potentially learning interaction structure. A GAM is interpretable because the `f(age)` term is readable. A KAN is interpretable because every edge function is readable *and* the composition structure tells you how features combine. KAN subsumes additive structure as a special case: a one-hidden-layer KAN with appropriate regularisation will converge to an approximately additive solution on problems where additivity holds.

**KAN vs NAMs:** Neural Additive Models (included in `insurance-gam.anam`) use separate sub-networks per feature and enforce additivity architecturally. They are a modern, neural implementation of the GAM constraint. KAN makes no additivity constraint.

**KAN vs SHAP relativities:** [SHAP relativities](/shap-relativities) explain any model post-hoc by decomposing predictions into feature contributions. They are an approximation of the model's behaviour, not the model itself. A KAN's edge functions are not an approximation of anything — they are the actual learned functional forms. The distinction matters if you need to put the model's logic in a governance document and attest to it.

**KAN vs distributional models (GAMLSS):** GAMLSS fits separate parametric models for mean and dispersion. KAN could replace the linear predictor structure in either model, but this combination has not been published. It is technically straightforward but unvalidated.

---

## A minimal implementation

The `pykan` library is pip-installable. The gap for actuarial use is the Poisson loss with exposure offset, which requires a custom loss function. Here is the pattern:

```python
import torch
import torch.nn as nn
from kan import KAN  # pip install pykan

# Assume X is a torch.Tensor of shape (n, p) - rating factors
# y is claim counts, exposure is policy years
# Both exposure and y are 1D tensors of shape (n,)

class PoissonKAN(nn.Module):
    def __init__(self, in_features: int, hidden: int = 5, grid: int = 5):
        super().__init__()
        self.kan = KAN(
            width=[in_features, hidden, 1],
            grid=grid,       # number of spline knots per edge
            k=3,             # spline order (cubic)
            seed=42,
        )

    def forward(self, x):
        # KAN output is unconstrained log-rate (before exp)
        return self.kan(x).squeeze(-1)


def poisson_deviance_loss(
    log_rate: torch.Tensor,
    y: torch.Tensor,
    log_exposure: torch.Tensor,
) -> torch.Tensor:
    """
    Poisson deviance with exposure offset.
    log_rate: model output (log link, no offset)
    y: observed claim counts
    log_exposure: log of policy years
    """
    mu = torch.exp(log_rate + log_exposure)
    # Poisson deviance: 2 * (y*log(y/mu) - (y - mu))
    # Numerically stable form:
    deviance = 2 * (
        torch.xlogy(y, y / mu.clamp(min=1e-10)) - (y - mu)
    )
    return deviance.mean()


# Training loop sketch
model = PoissonKAN(in_features=X_train.shape[1], hidden=5, grid=5)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    model.train()
    log_rate = model(X_train)
    loss = poisson_deviance_loss(log_rate, y_train, log_exposure_train)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# After training: visualise the learned edge functions
model.kan.plot()  # produces a grid of plots, one per edge

# Attempt symbolic regression on each edge
model.kan.auto_symbolic()
model.kan.print_symbolic()  # prints closed-form expressions where found
```

The `grid` parameter controls the number of spline knots. Start with 3–5. Increase if the model is underfitting; too many knots causes overfitting and slows symbolic regression matching. The `k=3` setting is cubic splines — the default and generally appropriate.

For monotone constraints on specific features (e.g., age, vehicle age, NCD), the MonoKAN implementation (available from the arXiv:2409.11078 supplementary code) replaces the KAN layer with a Hermite-spline layer and requires positive Hermite coefficients. The API is analogous: specify which input indices should be monotone increasing or decreasing before fitting.

---

## The honest performance picture

The benchmarking study by Poeta et al. (2024), "Benchmarking KANs" (arXiv:2406.14529), tested KANs on a range of tabular regression and classification tasks. The headline finding: KANs are competitive with MLPs on small tabular datasets but do not consistently beat XGBoost or LightGBM. Training time is 2–5x slower than comparable MLPs, which are themselves slower than gradient boosting on tabular data.

This is not a reason to dismiss KANs for insurance pricing. The comparison is wrong. KANs are not competing with XGBoost on predictive accuracy; they are competing with GAMs on the interpretability-accuracy frontier. A GAM (EBM) also loses on raw Gini to a deep GBM. The question is whether the interpretability gain is worth the accuracy cost, and for which use cases. The right comparison is:

- vs GLM: KAN almost certainly wins on Poisson deviance, at the cost of losing closed-form factor tables (unless symbolic regression succeeds)
- vs GAM/EBM: likely similar accuracy, KAN may generalise better on interaction structure, EBM trains faster
- vs GBM: GBM likely wins on Gini, KAN wins on auditability

The training speed issue is real for production workflows. A GBM on French MTPL (678,013 policies, 12 rating factors) trains in seconds. A KAN with `width=[12, 10, 1]` will take minutes. On a UK personal lines book of 2–3 million rows, this is a constraint. We would fit KANs on a stratified subsample for governance and documentation purposes, then use the symbolic regression output to inform the shape of explicit GLM terms.

---

## The governance angle

The symbolic regression step is where KAN potentially earns its place in a UK pricing workflow that is not accessible to any of the other architectures we cover.

Once a KAN is trained, `auto_symbolic()` attempts to match each edge function to a library of elementary functions. If the match exceeds a configurable similarity threshold, it records the symbolic form. A trained KAN on motor claim frequency might produce something like:

```
edge(ncd_years → hidden[0]): -0.31 * log(1 + ncd_years)
edge(driver_age → hidden[1]): 0.84 * exp(-0.04 * (age - 35)^2)
edge(vehicle_group → hidden[2]): 0.12 * vehicle_group
```

This is a closed-form expression for what the model learned. It is not a SHAP attribution — it is the model's actual weight functions, readable as mathematical expressions. For a model governance submission, "the model assigns log-linear NCD discounts with coefficient −0.31" is a statement of model structure. "SHAP values for NCD_years average −0.28 across the portfolio" is a statement about model behaviour on the training data. These are different things, and model governance under SS1/23 requires the former.

No other architecture in our stack produces this. EBMs produce lookup tables (close, but not symbolic). NAMs produce neural network weights (not symbolic). SHAP is post-hoc attribution. Only KAN, via symbolic regression, can output a closed-form mathematical expression for what it learned — and that expression can be inspected, challenged, and justified in a way that a neural network weight matrix cannot.

---

## What we think

KANs are not a replacement for GLMs in UK pricing. The production path for most firms remains: fit a GLM for the rate table, possibly with GAM-derived shape functions to inform bucketing decisions, run a GBM as a challenger model, extract SHAP relativities to check the GLM is not missing something important.

KANs fit a specific gap that none of those steps fill: you want a model that learns its own functional forms, can enforce certified monotonicity, and can produce a symbolic expression of what it learned — all in one object. That combination is useful for governance documentation, for mortality and longevity modelling (where the ASTIN Bulletin results are now published), and for complex non-life structures where you suspect the true link function is not log-linear but you do not want to guess what it is.

The non-life application — Poisson frequency, Gamma severity, exposure offsets, NCD and vehicle age monotonicity — is technically straightforward but has not been published with results on a standard actuarial dataset. That is the open work. We will run it on French MTPL and report back.

---

**References**

Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y. and Tegmark, M. (2024). KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756. *ICLR 2025.*

Xu, X., Badescu, A. and Pesenti, S. (2026). What KAN mortality say: smooth and interpretable mortality modeling using Kolmogorov-Arnold networks. *ASTIN Bulletin*, 56(1), pp. 32–59. DOI: [10.1017/asb.2025.10079](https://doi.org/10.1017/asb.2025.10079).

Polo-Molina, A., Alfaya, D. and Portela, J. (2024). MonoKAN: Certified Monotonic Kolmogorov-Arnold Network. arXiv:2409.11078.

Poeta, E., Giobergia, F., Pastor, E., Cerquitelli, T. and Baralis, E. (2024). A Benchmarking Study of Kolmogorov-Arnold Networks on Tabular Data. arXiv:2406.14529.
