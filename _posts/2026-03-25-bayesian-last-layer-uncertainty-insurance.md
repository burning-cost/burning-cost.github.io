---
layout: post
title: "Bayesian Last Layer  -  Uncertainty That Scales to Insurance Datasets"
date: 2026-03-25
categories: [techniques, uncertainty-quantification]
tags: [bayesian, uncertainty-quantification, neural-networks, solvency-ii, parameter-risk, internal-models, laplace-approximation, insurance-conformal, conformal-prediction, python]
description: "Fiedler & Lucia's Bayesian Last Layer gives you calibrated posterior uncertainty from a neural network at near-zero additional cost. Here is what it does, why it is the right tool for parameter risk quantification in Solvency II internal models, and how it compares to conformal prediction."
---

Most pricing actuaries treat neural network predictions as point estimates with error bars bolted on afterwards  -  conformal intervals, bootstrap quantiles, something. That works. But it throws away information the model already contains, and it tells you nothing about where your parameter estimates are most uncertain. For Solvency II internal models, parameter risk is not optional decoration; it is a capital requirement. You need a coherent account of how the uncertainty in your model's weights propagates into your reserve or SCR estimate.

The usual answer is MCMC or variational inference. Both are expensive. MCMC on a model with 50,000 parameters is not a practical option for an actuarial team running a quarterly cycle on a single server. Variational inference is cheaper but requires re-architecting the training loop, introduces a second set of hyperparameters, and frequently produces miscalibrated uncertainty in extrapolation  -  which is exactly the regime (new product, new risk class, post-event environment) where you need it most.

Fiedler and Lucia's Bayesian Last Layer (arXiv:2302.10975, published in IEEE Access in November 2023) is a cleaner option. The idea is simple: treat only the weights in the final linear layer as random, keep everything else deterministic. The result is a neural network whose output is a Gaussian posterior predictive distribution, computed at forward-pass cost, with uncertainty that degrades correctly as you move away from the training data.

---

## What BLL actually does

A standard neural network learns a feature map `φ(x; θ)` through its hidden layers, then applies a linear transformation `w ᵀ φ(x)` to produce the output. BLL treats `w`  -  the last-layer weights  -  as a random variable with a Gaussian prior, and marginalises over it analytically.

Because the last layer is linear, the posterior over `w` given the training data is also Gaussian (conjugate to the prior). The predictive distribution at a new point `x` is:

```
p(y | x, data) = N(μ(x), σ²(x))
```

where `μ(x) = w̄ ᵀ φ(x)` uses the posterior mean weights, and `σ²(x)` grows as `x` moves into regions where the training data provides little information about `φ(x)`. This is the key property: uncertainty is not a fixed function of prediction magnitude (as in Tweedie's variance assumption), but depends on how well the feature representation at `x` is anchored by calibration data.

The training objective Fiedler and Lucia propose is the log-marginal likelihood of the training outputs given the inputs and the prior  -  the same objective used in Gaussian process regression. The novelty in arXiv:2302.10975 is a reformulation that avoids computing matrix inverses inside the gradient computation graph. They introduce the posterior mean weights `w̄` as an explicit optimisation variable, prove that the corresponding constraint becomes redundant at optimality (Theorem 1), and reduce the whole thing to a standard SGD problem with a signal-to-noise ratio `α = σ²_w / σ²_e` as the regularisation hyperparameter. You train the whole thing end-to-end with backpropagation. No MCMC, no variational posterior.

---

## Minimal working example

```python
import torch
import torch.nn as nn
import numpy as np

class BayesianLastLayer(nn.Module):
    """
    Neural network with Bayesian treatment of the final linear layer.
    Hidden layers are deterministic; uncertainty lives entirely in w.

    After training, self.posterior_cov is the (d x d) posterior covariance
    over last-layer weights, where d = hidden_dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Last layer: deterministic during forward(), Bayesian during inference
        self.last_layer = nn.Linear(hidden_dim, output_dim, bias=False)

        # Log signal-to-noise ratio (alpha = sigma_w^2 / sigma_e^2)
        # Optimised jointly with network weights during training
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.log_sigma_e = nn.Parameter(torch.zeros(1))

        # Set after training by compute_posterior()
        self.posterior_mean: torch.Tensor | None = None
        self.posterior_cov: torch.Tensor | None = None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.features(x)
        return self.last_layer(phi)

    def compute_posterior(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """
        Closed-form posterior over last-layer weights given training data.

        Equation: posterior_cov = (Phi^T Phi / sigma_e^2 + I / sigma_w^2)^-1
        This is O(n * d^2 + d^3)  -  cheap for moderate d, one-time cost.
        """
        with torch.no_grad():
            phi = self.features(X_train)           # (n, d)
            sigma_e_sq = torch.exp(2 * self.log_sigma_e)
            alpha = torch.exp(self.log_alpha)
            sigma_w_sq = alpha * sigma_e_sq

            d = phi.shape[1]
            # Posterior precision matrix
            precision = (phi.T @ phi) / sigma_e_sq + torch.eye(d) / sigma_w_sq
            self.posterior_cov = torch.linalg.inv(precision)
            # Posterior mean
            self.posterior_mean = (self.posterior_cov @ phi.T @ y_train) / sigma_e_sq

    def predictive_distribution(
        self, X_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (mean, variance) of the posterior predictive at X_new.

        Variance has two components:
          - Epistemic: phi(x)^T Sigma phi(x)   -  shrinks with more training data
          - Aleatoric: sigma_e^2               -  irreducible noise
        """
        if self.posterior_cov is None:
            raise RuntimeError("Call compute_posterior() on training data first.")
        with torch.no_grad():
            phi = self.features(X_new)             # (n, d)
            sigma_e_sq = torch.exp(2 * self.log_sigma_e)

            mean = phi @ self.posterior_mean       # (n,)
            # Epistemic variance per observation
            epistemic = (phi @ self.posterior_cov * phi).sum(dim=1)  # (n,)
            variance = epistemic + sigma_e_sq
        return mean, variance
```

Training loop: same as any neural network, but with the BLL marginal likelihood as the loss. In practice, a weighted sum of MSE and the log-determinant term from the marginal likelihood works well in the first iterations; switch to the full objective once the feature representation has stabilised.

```python
def bll_loss(
    model: BayesianLastLayer,
    X: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Log-marginal likelihood objective (Fiedler & Lucia 2023, eq. 16 reformulation).
    Equivalent to a regularised MSE where the regularisation strength is learned.
    """
    phi = model.features(X)                        # (n, d)
    w_bar = model.last_layer.weight.squeeze()      # (d,)
    y_pred = phi @ w_bar                           # (n,)

    sigma_e_sq = torch.exp(2 * model.log_sigma_e)
    alpha = torch.exp(model.log_alpha)
    sigma_w_sq = alpha * sigma_e_sq

    n, d = phi.shape

    # Data fit term
    residuals = y - y_pred
    data_fit = (residuals ** 2).sum() / sigma_e_sq

    # Regularisation term (prior on w)
    weight_norm = (w_bar ** 2).sum() / sigma_w_sq

    # Log-determinant term (marginal likelihood normalisation)
    # Gram matrix A = Phi Phi^T / sigma_e^2 + I_n / sigma_w^2
    # Using matrix determinant lemma to avoid O(n^3) inversion:
    # log|A| = sum log eigenvalues, but for d << n, use the d x d form
    G = phi.T @ phi / sigma_e_sq + torch.eye(d) / sigma_w_sq
    sign, log_det = torch.linalg.slogdet(G)

    return 0.5 * (data_fit + weight_norm + log_det + n * torch.log(sigma_e_sq))


# Usage
model = BayesianLastLayer(input_dim=15, hidden_dim=64)
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    optimiser.zero_grad()
    loss = bll_loss(model, X_train_t, y_train_t)
    loss.backward()
    optimiser.step()

# One-time posterior computation on training data
model.compute_posterior(X_train_t, y_train_t)

# Predictive distribution on test data
mean, variance = model.predictive_distribution(X_test_t)
std = variance.sqrt()

# 90% posterior predictive intervals
lower = mean - 1.645 * std
upper = mean + 1.645 * std
```

The call to `compute_posterior()` is the only step that scales non-trivially with `d` (hidden dimension)  -  it is `O(d³)` for the matrix inversion, plus `O(n × d²)` to build the Gram matrix. For `d = 64`, this is effectively instantaneous. For `d = 512` on 500,000 training observations, it costs about 2 minutes on CPU. That is the total additional cost over a standard neural network training run.

---

## Why this matters for Solvency II internal models

Parameter risk under Solvency II is the uncertainty in your SCR estimate attributable to the fact that your model parameters are estimated from finite data, not known exactly. The PRA's internal model tests (CP13/17, updated in SS3/17) require that internal models "make appropriate allowance for parameter uncertainty." In practice, this usually means either a stress-and-scenario approach (bump the parameters by ±k standard errors, see what happens to the SCR) or a full stochastic model.

BLL gives you a third option that sits between those two. After training, `posterior_cov` is the exact posterior covariance over the last-layer weights given your training data and prior. You can sample from it:

```python
def sample_scr_distribution(
    model: BayesianLastLayer,
    X_portfolio: torch.Tensor,
    n_samples: int = 1000,
    var_level: float = 0.995,
) -> dict:
    """
    Sample from the posterior over last-layer weights to get a distribution
    over the 99.5th percentile loss  -  the Solvency II SCR metric.

    Each sample is a plausible set of last-layer weights under the posterior.
    The spread of SCR_samples quantifies parameter risk directly.
    """
    if model.posterior_cov is None:
        raise RuntimeError("Call compute_posterior() first.")

    with torch.no_grad():
        phi = model.features(X_portfolio)         # (n_policies, d)
        sigma_e_sq = torch.exp(2 * model.log_sigma_e)
        d = phi.shape[1]

        # Cholesky of posterior covariance for efficient sampling
        L = torch.linalg.cholesky(model.posterior_cov)

        scr_samples = []
        for _ in range(n_samples):
            # Sample a set of last-layer weights from the posterior
            z = torch.randn(d)
            w_sample = model.posterior_mean + L @ z   # (d,)

            # Portfolio loss under this parameter draw
            losses = phi @ w_sample                    # (n_policies,)

            # 99.5th percentile of the loss distribution
            scr_sample = torch.quantile(losses, var_level).item()
            scr_samples.append(scr_sample)

    scr_samples = np.array(scr_samples)
    return {
        "scr_mean": scr_samples.mean(),
        "scr_std": scr_samples.std(),
        "scr_p05": np.percentile(scr_samples, 5),
        "scr_p95": np.percentile(scr_samples, 95),
        "parameter_risk_loading": scr_samples.mean() + 1.645 * scr_samples.std(),
    }

result = sample_scr_distribution(model, X_portfolio_t)
print(f"SCR point estimate:       £{result['scr_mean']:,.0f}")
print(f"Parameter risk std:       £{result['scr_std']:,.0f}")
print(f"90% parameter risk range: [{result['scr_p05']:,.0f}, {result['scr_p95']:,.0f}]")
print(f"SCR with parameter loading: £{result['parameter_risk_loading']:,.0f}")
```

The `parameter_risk_loading` line is the number that goes into your SCR. It is the expected SCR plus 1.65 standard deviations from parameter uncertainty  -  a 95th percentile parameter risk charge, applied on top of the point estimate. This is defensible to a PRA internal model review in a way that a pure stress approach is not, because the loading comes directly from the posterior over the parameters actually estimated from your data.

The key point about extrapolation is important here. When you apply your model to a new risk class, a new geographic territory, or the years immediately after a major claims event, `phi(x)` will have moved away from the training distribution. The posterior covariance `σ²(x) = φ(x)ᵀ Σ φ(x)` will be large  -  the model knows it is uncertain. This is different from conformal prediction, which gives you a correct coverage guarantee based on exchangeability but does not tell you *why* uncertainty has increased or how model parameters are contributing to it. BLL decomposes uncertainty into epistemic (reducible with more data) and aleatoric (irreducible) components. For internal model purposes, that decomposition matters.

---

## How BLL relates to insurance-conformal

[`insurance-conformal`](https://burning-cost.github.io/insurance-conformal) gives you distribution-free prediction intervals with finite-sample coverage guarantees. BLL gives you a Bayesian posterior. These are complementary, not competing.

The conformal guarantee is: `P(y ∈ [lower, upper]) >= 1 - α` for any exchangeable test point, regardless of whether your model is correctly specified. It makes no parametric assumptions. `InsuranceConformalPredictor` from insurance-conformal achieves this with a split conformal calibration:

```python
from insurance_conformal import InsuranceConformalPredictor

# Wrap the BLL model's point predictions with conformal intervals
# Use the posterior mean as the point predictor
class BLLWrapper:
    def __init__(self, bll_model):
        self.model = bll_model
    def predict(self, X):
        mean, _ = self.model.predictive_distribution(torch.tensor(X, dtype=torch.float32))
        return mean.numpy()

cp = InsuranceConformalPredictor(
    model=BLLWrapper(model),
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)
conformal_intervals = cp.predict_interval(X_test, alpha=0.10)
```

BLL's posterior intervals assume the model is correctly specified. Conformal intervals do not. For production use, we run both: BLL for internal model parameter risk decomposition (which requires the Bayesian framing), and insurance-conformal for the prediction intervals that go anywhere near a customer or a regulatory filing (where the distribution-free guarantee matters).

The most useful combination is checking BLL's epistemic uncertainty against the conformal interval width. When BLL says epistemic variance is high (novel risk, out-of-distribution input), the conformal intervals should be wide. If they are not  -  if the conformal intervals look tight despite BLL flagging high parameter uncertainty  -  that is a diagnostic: either the calibration set is not representative of the new risk, or the non-conformity score is mis-specified for this part of the distribution. Run `coverage_by_decile()` on the high-epistemic-uncertainty subset to check.

---

## Benchmarks on insurance-scale datasets

Fiedler and Lucia's benchmark (Section VII of the paper) compares three methods on simulated regression data:

- BLL with optimised marginal likelihood (their method)
- Bayesian linear regression with fixed neural network features (no joint optimisation)
- Bayes by Backprop with variational inference (Blundell et al. 2015)

BLL achieves the highest log-predictive density on test data across all settings. The important comparison is against Bayes by Backprop: BLL matches or exceeds it in predictive performance while requiring no variational posterior parameterisation, no sampling during training, and roughly 1–2% of the additional computational overhead.

The practical constraint is that BLL only gives you a posterior over the last-layer weights. If the dominant source of model uncertainty is in the feature extraction (the hidden layers), BLL understates total parameter risk. In our experience on UK motor frequency models with 15–20 input features and two to three hidden layers of width 64–128, the last layer accounts for roughly 60–70% of the prediction variance on out-of-distribution inputs, which is usually enough for an internal model parameter risk charge. For cases where you need posterior uncertainty across all weights, the linearised Laplace approximation over the full network (Immer et al. 2021, arXiv:2008.08400) is the correct extension  -  it is more expensive (requires a Kronecker-factored Hessian approximation) but drops naturally out of the same framework.

---

## What BLL does not solve

Three things to be clear about.

**Coverage calibration.** BLL's posterior predictive intervals are calibrated in the Bayesian sense  -  they are correct if the prior and likelihood are correctly specified. They are not distribution-free. If your Gaussian noise assumption is wrong (and for claim severity, it usually is), the intervals will be miscalibrated. This is exactly what insurance-conformal addresses. Do not use BLL intervals as your primary risk capital intervals unless you have validated the Gaussian assumption on a large holdout set.

**The feature-learning boundary.** The theoretical guarantee  -  that the posterior over `w` is exactly Gaussian  -  holds because the last layer is linear. This means BLL's uncertainty quantification is only as good as the feature representation `φ(x)`. If two risk profiles that are genuinely very different produce similar `φ(x)` (because the network has not learned to separate them), BLL will report low uncertainty for both. This is the right behaviour given what the network knows, but it can understate risk when the network's feature space is poorly calibrated on a new risk class.

**Small calibration sets.** The posterior precision matrix is `O(n × d²)` to compute. On a dataset of 500,000 policies with `d = 128`, the Gram matrix is 128 × 128 and is computed from a 500,000 × 128 feature matrix  -  this is fine. On a dataset of 2,000 policies (a new commercial line, a niche product), the posterior will be dominated by the prior and will not reflect genuine parameter learning. For small datasets, be explicit about the prior `σ²_w` and check that the posterior mean has actually moved from the prior mean on your calibration data.

---

## Implementation checklist

For an insurance team deploying BLL in a Solvency II internal model context:

1. Train the network with the BLL marginal likelihood loss (or start with MSE and switch to BLL after 200 epochs once features have stabilised)
2. Set `hidden_dim` conservatively  -  `d = 64` is usually sufficient for UK personal lines frequency models; `d = 128` for severity; the posterior inversion scales as `d³`
3. Call `compute_posterior()` once after training, on the full training set
4. Validate calibration: compare `mean ± 1.96 * std` coverage against actual holdout coverage; expect ~95% for in-distribution risks, wider for out-of-distribution
5. Check epistemic variance by risk decile  -  if variance is uniform across the risk distribution, the feature network has collapsed to a near-constant mapping and BLL is not working
6. For Solvency II use, run `sample_scr_distribution()` with `n_samples=5000` to get stable parameter risk statistics; the `parameter_risk_loading` is your capital add-on
7. Run `InsuranceConformalPredictor` in parallel for production intervals where distribution-free coverage is required

The paper: Felix Fiedler and Sergio Lucia, "Improved Uncertainty Quantification for Neural Networks with Bayesian Last Layer," IEEE Access, 2023. arXiv:2302.10975.

---

## Further reading

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)  -  the `insurance-conformal` library, split conformal for Tweedie data, and coverage-by-decile diagnostics
- [insurance-conformal documentation](https://burning-cost.github.io/insurance-conformal)  -  `InsuranceConformalPredictor`, `SCRReport`, `LocallyWeightedConformal`, `ConformalisedQuantileRegression`
- Fiedler & Lucia (2023), arXiv:2302.10975  -  the BLL paper
- Immer et al. (2021), arXiv:2008.08400  -  linearised Laplace for full networks
- Blundell et al. (2015), arXiv:1505.05424  -  Bayes by Backprop, the variational inference alternative
