---
layout: post
title: "Embedded Insurance Pricing and the API Underwriting Model"
date: 2026-03-25
categories: [pricing, architecture, regulation]
tags: [embedded-insurance, api-pricing, glm, latency, distillation, consumer-duty, fca, fair-value, api-underwriting, insurance-distill, insurance-governance, real-time-pricing, uk-personal-lines]
description: "BCG's 2025 analysis puts embedded insurance at 30% CAGR. The pricing architecture question is not whether to do it - it's whether your model can answer in under 100ms without compromising actuarial integrity. Here is the engineering and the governance."
---

Embedded insurance - the product appearing at point of sale inside a partner's digital journey rather than through a standalone insurer or broker channel - is growing fast by any measure. BCG's 2025 embedded finance analysis puts the segment at 30% compound annual growth, with personal lines the dominant volume driver: travel insurance inside booking flows, device protection at e-commerce checkout, gap and mechanical breakdown cover inside motor dealer finance journeys. The FCA's 2024 supervisory review of BNPL and embedded financial products flagged it as a priority watch area.

The commercial opportunity is real. The actuarial and engineering challenge that nobody discusses honestly is also real: at embedded distribution scale, your pricing model is an API endpoint, not a monthly batch process. It needs to return a premium in under 100 milliseconds. It needs to do that 50,000 times per day, often under burst traffic conditions at peak retail periods. And it needs to do all of that while satisfying Consumer Duty fair value obligations and PRA model governance expectations that were written for entirely different operating conditions.

This post is about the architecture required and, specifically, about the distillation techniques that make a production-grade actuarial GLM fast enough to serve.

---

## The latency budget

A consumer checkout flow requires the full page to render in under two seconds to avoid material cart abandonment. In practice, the insurer's quote API has a budget of 80-120ms - the rest is the partner's front-end, payment processing, and network round-trip. At 100ms, you have approximately:

- 10ms for network ingress and authentication
- 20ms for input validation and feature derivation
- 50ms for model inference
- 10ms for output formatting and logging
- 10ms buffer

Fifty milliseconds for model inference. That is enough time for a pre-loaded scikit-learn GLM to score tens of thousands of observations. It is not enough time for anything that involves:
- Loading model artefacts from cold storage on each request
- Calling an external scoring service over HTTP
- Running a CatBoost model with 500 trees and deep interactions on a single CPU thread
- Recomputing partial dependences at serving time

The common failure mode is that pricing teams build an excellent CatBoost model in development - good Gini, clean validation, proper out-of-time splits - then serve it directly via an ML platform that adds latency they never measured during development. The model that takes 2 seconds to score in a batch environment takes 2 seconds in the API too, unless someone explicitly addresses it.

---

## Why you cannot just serve a GBM directly

The temptation to serve a CatBoost or LightGBM model directly is understandable. You have it. It works. In a containerised microservice with a pre-loaded model binary, CatBoost can score a single observation in 2-5ms on modern hardware. So where is the problem?

Three places.

**Interpretability under Consumer Duty.** Under PRIN 2A and the FCA's Consumer Duty guidance (FG22/5, July 2022), firms must be able to explain how a price was arrived at if asked. A GBM with 800 trees and SHAP explanations computed on demand is defensible under a proportionality argument, but the FCA has repeatedly indicated that it expects pricing to be explainable in terms customers can understand. A multiplicative GLM with explicit rating factors for age, occupation, declared value, and term is structurally transparent in a way that SHAP attributions on a GBM are not. This matters more in embedded channels, where the customer has less context and the FCA has explicitly flagged that complexity is a fair value concern.

**Regulatory documentation requirements.** PRA SS1/23 (the supervisory statement on model risk management) was written for banks, not insurers. However, the FCA's Model Risk Management guidance (MS23/1) and PRA supervisory expectations under Solvency II mean that insurers face materially equivalent governance obligations in practice. The `RiskTierScorer` in insurance-governance uses a tier framework aligned with these expectations (for the full framework, see [PRA SS1/23-compliant model validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)). Many insurers are voluntarily adopting the SS1/23 principles as a baseline. A GBM served as a black box in an API endpoint is a Tier 1 (Critical) model — 60+ points on the `RiskTierScorer`, annual review, Model Risk Committee sign-off. A GLM surrogate, with documented factor tables and an explicit validation report showing its deviation from the GBM parent, can sometimes be structured as a Tier 2 implementation of the same business logic. The tier matters because it determines the review cycle and sign-off authority, which in turn determines how quickly you can deploy changes.

**Maintenance overhead.** GLM factor tables can be updated in a rating engine (Radar, Emblem, a bespoke table loader) without a full model deployment cycle. When your GBM changes - new training data, feature engineering update - the API deployment needs to go through a full model change process. When a factor table updates, it can go through a simpler rate change process. At embedded distribution volumes, you will be updating rates frequently. The operational overhead of treating every rate update as a model deployment is substantial.

---

## Distillation for API serving

The right architecture for most embedded insurance pricing models is a GBM fitted on all available data, then distilled into a multiplicative GLM that runs in the API. The GBM provides predictive power. The GLM provides speed, interpretability, and governance tractability. The question is how much predictive performance you lose in the distillation, and whether that loss exceeds the commercial benefit of embedding.

[insurance-distill](/insurance-distill/) implements two distillation approaches. `SurrogateGLM` fits a GLM on GBM pseudo-predictions with optimal binning. `LassoGuidedGLM` uses the GBM's partial dependence curves to place bin boundaries, then uses lasso to select which bins are large enough to matter. Both produce multiplicative factor tables that load directly into a rating engine.

For embedded insurance specifically, `LassoGuidedGLM` is usually the better choice because the feature space is typically thin - embedded products have limited data collection at checkout - and the lasso selection prevents overfitting to the available features:

```python
import polars as pl
from insurance_distill import LassoGuidedGLM

lg = LassoGuidedGLM(
    gbm_model=fitted_catboost,
    feature_names=["customer_age", "declared_value", "purchase_category", "term_days"],
    n_bins=8,
    alpha=0.5,       # lasso regularisation — tighten if feature space is thin
    family="gamma",  # severity model
)
lg.fit(X_train, y_train, exposure=exposure_arr)

tables = lg.factor_tables()
# Returns dict: feature -> DataFrame with columns [bin_label, factor]
# e.g. {'customer_age': shape (8, 2), 'declared_value': shape (6, 2), ...}

lg.summary()
# Prints: Gini retention, deviance ratio, selected bins per feature
```

The `factor_tables()` output is the artefact that goes into your API. Load it at startup, keep it in memory, apply factors multiplicatively with the base rate. At that point, the scoring path in the API is: feature lookup → multiply three or four factors → return. Sub-millisecond. No model library needed at serving time.

For `SurrogateGLM`, the workflow is similar but the binning uses a decision tree on GBM predictions rather than PD curves:

```python
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=fitted_catboost,
    X_train=X_train,          # Polars DataFrame
    y_train=y_train,
    exposure=exposure_arr,
    family="poisson",          # frequency model
)
surrogate.fit(max_bins=10)

report = surrogate.report()
print(report.metrics.summary())
# Gini: GBM=0.412, GLM surrogate=0.387 (94% retention)
# Deviance ratio: 0.91

surrogate.export_csv("output/embedded_freq_factors/")
```

The `report.metrics.summary()` output is what you present to the Model Risk Committee to justify the distillation. A Gini retention of 94% with deviance ratio of 0.91 is a strong result for a four-feature embedded product; you would expect lower retention on a full personal lines motor book where the GBM is capturing deep interactions across dozens of features.

---

## What Consumer Duty means for embedded fair value

The FCA's Consumer Duty guidance specifically discusses embedded and ancillary insurance under the fair value rules. The key obligation under PRIN 2A.4 is that the price paid by the customer must be reasonable relative to the overall benefits of the product. For embedded insurance, this creates a specific analytical challenge: the customer is usually not comparing your embedded product against alternatives at point of sale. The comparison set is implicit, not explicit.

The FCA's 2023 review of add-on insurance (which substantially overlaps with what is now called embedded insurance) found that add-on products showed loss ratios materially below comparable standalone products - sometimes by 20-30 percentage points. The interpretation is straightforward: the convenience premium at checkout was being captured as insurer profit rather than returned to customers as value. Consumer Duty makes that explicit standard unlawful.

For pricing teams, the practical obligation is to document that the embedded price reflects actuarial cost plus a reasonable commercial loading, and that the commercial loading is not materially higher than in equivalent standalone products. That documentation needs to be model-governance-grade - not an ad-hoc analysis, but part of the model's ongoing validation record.

[insurance-governance](https://burning-cost.github.io/insurance-governance) provides the infrastructure for this. The `MRMModelCard` captures model purpose, data sources, assumptions, and limitations in a structured form. For an embedded pricing model, the card's assumptions section should explicitly include the fair value assumption - that the distilled GLM's predicted loss cost is a reasonable proxy for the GBM's prediction, and that the commercial loading above that loss cost is consistent with standalone product margins:

```python
from insurance_governance import MRMModelCard, Assumption

card = MRMModelCard(
    model_id="embedded-gadget-severity-v2",
    model_class="pricing",
    description="Gamma GLM severity model for embedded gadget insurance, "
                "distilled from CatBoost v7 trained on 2023-2025 claims.",
    champion_status="champion",
    distribution_family="Gamma",
    owner="Embedded Pricing Team",
    business_unit="Direct & Partnerships",
)

card.assumptions.append(Assumption(
    description="Distilled GLM retains >=90% of GBM Gini (94% on 2024 OOT val set). "
                "Residual 6% loss is in segment interactions not captured in factor table.",
    risk="MEDIUM",
    mitigation="Quarterly comparison of GLM vs GBM predictions on live traffic; "
               "trigger recalibration if Gini retention drops below 88%.",
    rationale="94% retention acceptable for embedded channel given interpretability "
              "and governance benefit; would not accept for full personal lines motor.",
))

card.assumptions.append(Assumption(
    description="Embedded channel loss ratio consistent with standalone equivalent "
                "within +/-5pp (current gap: +2pp). Consumer Duty fair value compliant.",
    risk="HIGH",
    mitigation="Half-yearly fair value assessment comparing embedded vs standalone "
               "loss ratios; escalation to Chief Actuary if gap exceeds 5pp.",
    rationale="FCA Consumer Duty PRIN 2A.4 requires documented evidence of fair value.",
))
```

The `risk="HIGH"` on the fair value assumption is deliberate. It is not a high-probability risk - we are not saying fair value is likely to fail - but the consequence of it failing silently is regulatory enforcement, and the `MRMModelCard` design treats consequence severity as the driver of risk rating, not probability alone. The mitigation makes the monitoring obligation explicit and traceable.

---

## The regulatory tier question

An embedded pricing model in production has high materiality (GWP impact), customer-facing regulatory exposure, and typically a non-trivial complexity score. Running `RiskTierScorer` on a realistic profile will return Tier 1 almost every time:

```python
from insurance_governance import RiskTierScorer

scorer = RiskTierScorer()
result = scorer.score(
    gwp_impacted=4_200_000,      # £4.2m GWP through embedded channel
    n_features=4,
    uses_external_data=False,
    last_validation_days=45,     # validated 45 days ago
    drift_triggers_12m=1,        # one monitoring trigger in past year
    in_production=True,
    customer_facing=True,
    regulatory_use=False,
)
print(result.tier, result.score, result.rationale)
# Tier 1 (Critical)  67  'GWP >£1m (25pts); production+customer-facing (25pts); ...'
```

Tier 1 means annual review and Model Risk Committee sign-off. That is the correct governance standard for an embedded pricing model at this scale. Trying to structure around it by splitting into smaller models or reducing documented GWP impact is governance theatre; any reviewer at the FCA who understands the channel will see through it.

What Tier 1 does not mean is that every rate update goes to the MRC. The model tier governs the model - the algorithm, the training data, the feature set, the validation methodology. Rate changes within the model - updating the factor tables from a new training run using the same methodology - can be handled via the rating change governance process, which is typically lighter. The distillation architecture specifically enables this: the GBM is the model (Tier 1, annual MRC review), and the factor tables are the rate (quarterly update via Chief Actuary sign-off).

---

## Practical architecture for embedded API pricing

The architecture we recommend for embedded insurance API pricing has four layers:

**Layer 1 - Batch GBM refit.** Run monthly or quarterly on the full training dataset. This is the actuarially serious model. CatBoost, full feature set, proper cross-validation. It does not run in the API; it lives in your modelling environment.

**Layer 2 - Distillation and factor export.** After each GBM refit, run `LassoGuidedGLM` or `SurrogateGLM` to produce factor tables. Run the validation suite (`report.metrics.summary()`, double-lift chart). Check Gini retention. If retention is above threshold, export factors via `surrogate.export_csv()`. This is a change to the factor tables, not to the model.

**Layer 3 - API factor loader.** A thin serving layer that loads factor tables from a configuration store (S3, a database, a feature store) at startup and applies them multiplicatively. No ML library required. This process restarts and reloads tables in under 2 seconds. Factor table updates are a configuration push, not a deployment.

**Layer 4 - Governance and monitoring.** `MRMModelCard` updated on each GBM refit. `ModelValidationReport` generated on each distillation run. Gini retention tracked over time. `RiskTierScorer` run annually or after material changes. These artefacts live in your model registry alongside the model binaries.

The latency at Layer 3 is sub-millisecond per request. The governance overhead of the GBM refit at Layer 1 is fully separated from the operational rate change at Layer 2. Consumer Duty fair value evidence is generated automatically as part of the distillation validation. The MRC sees the model at annual review; the Chief Actuary sees the factor tables at the quarterly rate change.

This is not a novel architecture. It is the standard multiplicative rating engine pattern that actuaries have been using for thirty years, upgraded to take advantage of machine learning at the fitting stage. The novel piece is the formalisation of the distillation step and the governance trail that connects the GBM to the rate tables. Without that trail, you have either a black-box API or an arbitrary GLM. With it, you have a defensible embedded pricing model.

---

**Related tools:** [insurance-distill](https://burning-cost.github.io/insurance-distill) - GBM to GLM distillation via surrogate fitting and partial-dependence-guided binning. [insurance-governance](https://burning-cost.github.io/insurance-governance) - model cards, risk tier scoring, and validation reports aligned with PRA supervisory expectations and Consumer Duty.

**References:** BCG Embedded Finance Report 2025; FCA Consumer Duty Guidance FG22/5 (July 2022); FCA Model Risk Management Discussion Paper MS23/1; PRA SS1/23 (banks — principles voluntarily adopted by many insurers); PRA Solvency II supervisory expectations; Lindholm & Palmquist (2024) SSRN 4691626.
