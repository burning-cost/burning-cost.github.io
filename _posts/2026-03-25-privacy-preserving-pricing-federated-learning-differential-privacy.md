---
layout: post
title: "Privacy-Preserving Pricing: Federated Learning and Differential Privacy"
date: 2026-03-25
categories: [techniques, regulation]
tags: [federated-learning, differential-privacy, uk-gdpr, pcw, data-pooling, insurance-fairness, fairness, pricing, python]
description: "UK GDPR constrains what pricing data you can share across entities. Federated learning and differential privacy offer a way around the constraint — but only if you understand where the privacy guarantees actually break down."
---

The industry's approach to shared data has a structural contradiction at its centre. PCW data pools — where price comparison websites aggregate shopping behaviour across multiple insurers — are enormously valuable for modelling conversion and demand elasticity. They are also, under UK GDPR Article 6, legally awkward. Every insurer that contributes data to a shared pool is processing personal data under a legal basis that may not transfer cleanly to a joint model training exercise run by a third party.

This matters because multi-insurer models are better. A single carrier training a conversion model on its own quotes sees selection bias from its own pricing — it never quotes cheaply enough to win on certain risks, so it never learns what drives conversion for those risks. A pooled model trained across five carriers sees the full market. The accuracy improvement is not marginal: pooled datasets that cover 60-70% of the UK motor market produce conversion lift that single-carrier models cannot replicate regardless of feature engineering.

The privacy-preserving answer is federated learning with differential privacy. The core idea has been around since 2017 (the Google Keyboard paper), but the insurance-specific treatment is newer. The ASTRI whitepaper (November 2025) and the NeurIPS 2025 f-DP result put numbers on the privacy-accuracy trade-off in settings that map directly to PCW data pools.

---

## What UK GDPR actually says

The relevant constraint is not that data cannot be shared — it is that processing must have a lawful basis, a specified purpose, and that purpose must be compatible with the original collection purpose. Insurers collect customer data for underwriting. Using that data to train a joint model for a PCW consortium is a new processing purpose. Article 6(4) requires a compatibility assessment. Legitimate interests (Article 6(1)(f)) require a balancing test. Special category data — anything that can function as a proxy for protected characteristics — requires explicit consent or a Schedule 2 DPA 2018 condition.

The ICO's guidance on data sharing (updated November 2024) is explicit that pseudonymisation does not make personal data anonymous. A quote record containing vehicle details, postcode, age band, and NCD level is re-identifiable given any of the source CRM systems. The practical implication: sharing raw quote records with a joint modelling entity exposes every contributing insurer to a processing activity that needs its own legal basis, purpose limitation assessment, and controller-processor agreement.

Federated learning sidesteps this. Model training happens locally within each insurer's environment. Only gradient updates — derivatives of the loss function with respect to model parameters — leave the site. No individual records cross the boundary.

---

## The federated learning setup for PCW data

A typical PCW conversion model has the following structure:

- **Features**: quote price, market rank (1st/2nd/3rd/4th+), vehicle group, age band, NCD, cover type, duration, voluntary excess, historical conversion rate for similar risks
- **Outcome**: did the customer buy? Binary. Logistic family.
- **Hierarchy**: carrier-specific conversion rates exist — Direct Line converts differently from Admiral on the same risk, holding rank constant, because brand and trust interact with price sensitivity

The federated version trains a shared global model across N carriers. Each carrier runs local gradient descent for T steps using its own data. Gradient updates are aggregated centrally (by the PCW, or by a trusted aggregator) and sent back. After R rounds, the global model has absorbed information from all carriers without any carrier's individual records leaving its systems.

In PyTorch this is straightforward:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class ConversionModel(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def local_update(
    global_weights: dict,
    X_local: torch.Tensor,
    y_local: torch.Tensor,
    n_epochs: int = 3,
    lr: float = 0.01,
) -> dict:
    """Run T steps of SGD on local data; return updated weights."""
    model = ConversionModel(n_features=X_local.shape[1])
    model.load_state_dict(global_weights)
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    loader = DataLoader(
        TensorDataset(X_local, y_local), batch_size=256, shuffle=True
    )
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimiser.zero_grad()
            criterion(model(xb), yb).backward()
            optimiser.step()

    return model.state_dict()


def federated_average(weight_list: list[dict], n_samples: list[int]) -> dict:
    """FedAvg: weighted mean of client weight updates."""
    total = sum(n_samples)
    avg = {}
    for key in weight_list[0]:
        avg[key] = sum(
            w[key] * (n / total) for w, n in zip(weight_list, n_samples)
        )
    return avg
```

The aggregation step is `federated_average`: a weighted mean of each carrier's local model weights, where the weight is proportional to the number of local training samples. This is FedAvg (McMahan et al., 2017). It converges to the same solution as centralised training under i.i.d. data. With non-i.i.d. data — which is the realistic case, because carrier books are not exchangeable — convergence is slower and requires more rounds.

---

## Where the privacy guarantee breaks down

Federated learning alone does not provide privacy in the formal sense. Gradient updates leak information about training data. The leakage is not theoretical: membership inference attacks (Shokri et al., 2017) can determine whether a specific record was in the training set with accuracy meaningfully above 50% from gradients alone. Model inversion attacks can reconstruct approximate input features.

Differential privacy (DP) is the formal solution. The definition: a randomised mechanism M satisfies (ε, δ)-DP if for any two adjacent datasets D and D' differing in one record, and any output set S:

```
Pr[M(D) ∈ S] ≤ exp(ε) · Pr[M(D') ∈ S] + δ
```

ε is the privacy budget. Lower ε means stronger privacy. δ is the failure probability — typically set at 1/n where n is the dataset size. The mechanism: clip gradients to a maximum norm C, then add Gaussian noise scaled to σ = C · √(2 ln(1.25/δ)) / ε before sharing.

In the NeurIPS 2025 f-DP work (Dong, Roth & Su), the guarantee is tightened using the f-divergence framework. Classical (ε, δ)-DP uses a crude union bound that overstates the privacy cost of composition — running DP training for R rounds. The f-DP result gives tighter composition bounds: the actual privacy cost of R rounds of DP-SGD with noise σ is:

```
ε_total(R) ≈ σ⁻¹ · √(2R · ln(1/δ))
```

rather than the naïve R·ε from sequential composition. For 50 rounds with σ=1.0 and δ=1e-5, f-DP gives ε_total ≈ 2.4 versus the naïve bound of 50·ε_per_round ≈ 18.5. This is not a marginal improvement — it is the difference between a viable and an unviable privacy budget.

The practical implementation adds Gaussian noise to gradients before the FedAvg step:

```python
import torch

def dp_clip_and_noise(
    weights_before: dict,
    weights_after: dict,
    clip_norm: float,
    noise_multiplier: float,
) -> dict:
    """
    DP-SGD gradient perturbation.

    Clips the per-client update norm to clip_norm, then adds
    Gaussian noise scaled to noise_multiplier * clip_norm.
    """
    noised = {}
    for key in weights_before:
        delta = weights_after[key] - weights_before[key]
        # Clip
        norm = delta.norm(2)
        delta = delta * min(1.0, clip_norm / (norm + 1e-8))
        # Add noise
        delta = delta + torch.randn_like(delta) * (noise_multiplier * clip_norm)
        noised[key] = weights_before[key] + delta
    return noised
```

The noise multiplier σ of 1.0 is the conventional starting point. It corresponds to ε ≈ 2.5 after 50 rounds with δ=1e-5, which is within the range ICO guidelines treat as adequately privacy-preserving for statistical outputs (the ICO DP guidance from January 2024 references ε ≤ 10 as a reasonable upper bound for low-risk processing).

---

## The accuracy cost

This is where honesty is required. DP degrades model accuracy. The noise injection that provides the formal privacy guarantee also injects noise into gradient updates. The ASTRI whitepaper (November 2025) benchmarks this on a synthetic multi-insurer pricing dataset with parameters calibrated to Hong Kong motor, but the structural results transfer.

Their findings at ε=2.0, 5 clients, 100 training rounds:

| Setting | AUC | Brier Score |
|---|---|---|
| Centralised (no privacy) | 0.812 | 0.187 |
| Federated (no DP) | 0.798 | 0.192 |
| Federated + DP (ε=2.0) | 0.771 | 0.208 |
| Federated + DP (ε=10.0) | 0.791 | 0.194 |
| Single carrier (no pooling) | 0.763 | 0.214 |

The key comparison is the bottom row against the DP rows. Even with ε=2.0 — a meaningfully tight privacy budget — the federated DP model beats a single-carrier model without pooling. The case for federated learning is that the accuracy gain from seeing the full market is larger than the accuracy loss from DP noise. At ε=10.0, the gap versus centralised narrows to 2 AUC points.

The practical implication for UK motor: if you are a mid-tier carrier training a conversion model on 800,000 quotes per year, your single-carrier model is operating with less data than the privacy-degraded federated model that sees 5 million quotes. The federated model wins even under strong privacy constraints.

---

## What the fairness library gives you

The [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) library's `PrivatizedFairnessAudit` module handles a related but distinct problem: pricing when the protected attribute (gender, age) has been privatised via local differential privacy before the data was shared. The MPTP-LDP protocol from Zhang, Liu & Shi (2025) applies.

The practical scenario: a carrier that has collected gender data under the pre-2012 framework must now train gender-neutral models but cannot simply drop gender — it needs to use a noise-corrected reweighting to ensure its models are discrimination-free without having access to clean sensitive attribute labels.

```python
from insurance_fairness import PrivatizedFairnessAudit

# S is the privatised gender attribute: 0/1 with known LDP noise
# epsilon=2.0 means the randomised response protocol used epsilon=2.0
audit = PrivatizedFairnessAudit(
    n_groups=2,
    epsilon=2.0,                       # LDP budget applied at data collection
    reference_distribution="uniform",  # equal group weighting for UK gender neutrality
    loss="poisson",
    nuisance_backend="catboost",
)
audit.fit(X_train, y_frequency, S_privatised, exposure=exposure)

# Discrimination-free premium predictions
fair_premium = audit.predict_fair_premium(X_test)

# Audit result includes the noise amplification factor C1
# and Theorem 4.3 generalisation bound
report = audit.audit_report()
print(f"pi (correct-response prob): {report.pi_estimated:.3f}")
print(f"Generalisation bound (95%): {report.bound_95:.4f}")
print(f"Negative weight fraction:   {report.negative_weight_frac:.3f}")
```

The `negative_weight_frac` diagnostic matters: if more than 5% of the reweighted observations have negative weights before clipping, the LDP noise is too heavy for reliable correction and the resulting model carries systematic bias. With ε=2.0 and binary groups, this is rarely a problem (C1 ≈ 2.6), but at ε=0.5 it becomes serious.

---

## What a pricing team actually needs to change

The gap between theory and practice in this area is wider than in most of our technical posts. Here is what implementing federated learning for a PCW data pool actually requires:

**Legal:** A data sharing agreement that explicitly scopes the processing to gradient aggregation, not record transfer. The controller-processor relationship needs to specify that the PCW (or aggregator) receives only noise-perturbed gradient updates and retains no carrier-identifiable information. This is not the same agreement as the existing PCW data sharing arrangement.

**Technical:** A trusted execution environment or secure multi-party computation framework for the aggregation step. FedAvg with DP is not sufficient on its own — a malicious aggregator can still recover information from gradient updates unless the aggregation itself happens in a trusted environment.

**Governance:** The privacy budget ε needs to be tracked across training rounds and refreshes. Once a model has been trained and refreshed 10 times with ε=3.0 per round, the cumulative privacy cost under composition rules is not 30. Under f-DP it is approximately 3.0 · √(10 · 2 · ln(1/δ)) ≈ 11.2 for δ=1e-5. This needs to be tracked by someone who understands the accounting.

**Model:** The federated model needs to handle carrier-specific heterogeneity. The naive FedAvg approach treats all carriers equally. A carrier with 80% of the quotes dominates. Weighted aggregation by quote volume is the obvious fix, but it does not handle differences in carrier book composition — a carrier that predominantly quotes sports cars has different gradient dynamics than one focused on standard private car.

The research is mature enough that the ICO and FCA have both engaged with it. The ICO's January 2024 guidance explicitly discusses federated learning as a privacy-enhancing technology. The FCA's AI sandbox (February 2026 update) includes federated model training as an eligible use case. The tooling is not the bottleneck. The legal and governance infrastructure around it is.
