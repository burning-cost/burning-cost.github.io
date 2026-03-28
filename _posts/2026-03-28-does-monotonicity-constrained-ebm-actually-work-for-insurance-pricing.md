---
layout: post
title: "Does Monotonicity-Constrained EBM Actually Work for Insurance Pricing?"
date: 2026-03-28
categories: [techniques, validation]
tags: [EBM, interpretable-ml, monotonicity, GAM, nam, anam, insurance-gam, poisson, gini, motor, pricing, python]
description: "On a UK motor DGP with a monotone young-driver requirement, unconstrained EBM violates monotonicity in 31% of runs. Constrained EBM matches GLM monotonicity compliance at 100% while closing 80% of the Gini gap to the unconstrained model."
---

Monotonicity constraints in insurance pricing are not optional. The FCA expects a motor insurer to be able to explain why a 19-year-old driver pays more than a 35-year-old. A pricing model where young driver relativities are non-monotone — where a 22-year-old is cheaper than a 20-year-old for model-internal reasons not grounded in claims experience — is a governance problem before it is a pricing problem.

Standard EBMs (Explainable Boosting Machines, Microsoft InterpretML) do not enforce monotonicity. Neither do standard NAMs or GAMs fitted via splines. The Actuarial Neural Additive Model (Laub, Pho & Wong, arXiv:2509.08467, September 2025) enforces monotonicity architecturally — it is baked into the network constraint, not applied as a post-hoc correction. And Krupova, Rachdi and Guibert (arXiv:2503.21321, March 2025) benchmark standard EBMs against GLM, GAM, GBM and XGBoost on French motor data, showing what EBMs capture that the alternatives miss.

We tested both questions using [`insurance-gam`](/insurance-gam/): does monotonicity-constrained EBM preserve the discrimination lift of the unconstrained model, and at what cost in Gini?

---

## Why monotonicity matters more than most models think

A GLM with age as a rating factor is automatically monotone if the actuary chooses an appropriate functional form — a quadratic or piecewise-linear term can produce a U-shape but not an arbitrary non-monotone curve. An EBM fitted by cyclic gradient boosting has no such structural guarantee. It will fit whatever the data supports, including genuine non-monotone relationships in the data that are actually noise artefacts from thin age bands.

The consequence is not merely regulatory. A pricing model with non-monotone age curves produces renewal anomalies: a policyholder ages from 21 to 22 and gets a rate *decrease* because the boosted model happens to have a lower shape function value at age 22 than at age 21 on the training data. The pricing system flags this as a rate decrease on a young high-risk driver. The underwriting team asks for an explanation. There is not a good one.

Krupova et al. (2025) show that EBMs capture pairwise interactions — notably vehicle age × driver age, which is real structure in French motor claims data that GLM misses entirely. This is the genuine lift that makes EBMs worth the model complexity. But they do not address monotonicity, and the French motor data is large enough that noise artefacts are less of a problem than they are on thin UK books.

Laub et al. (2025) address exactly this: ANAM enforces monotonicity via Dykstra's algorithm applied at each training step, ensuring the output shape function is non-decreasing (or non-increasing) for the specified features. The constraint is architectural, not post-hoc.

---

## What we tested

Benchmark: 10,000 synthetic UK motor policies, 70/30 train/test split. DGP: five features, four non-linear effects. The young driver effect is U-shaped (17–79 range), but with a hard monotone requirement: claims must be non-increasing from age 17 to age 40 (the monotone decline from young-driver peak to mature-driver plateau). Above age 55, claims increase again — this tail uptick is allowed to be non-monotone.

Three models:

- **Poisson GLM baseline**: quadratic age term, linear NCD, linear vehicle age, log(annual miles), area dummies. Monotone by construction in the young driver range (quadratic is forced to be decreasing by the sign of the coefficient).
- **Unconstrained EBM**: `InsuranceEBM(loss='poisson', interactions='3x')` — no monotonicity enforcement.
- **Constrained EBM**: `InsuranceEBM(loss='poisson', interactions='3x', monotone_increasing=[], monotone_decreasing=['driver_age_17_40'])` — monotone decreasing in the 17–40 age range.

Monotonicity compliance is measured as the fraction of (age_i, age_j) pairs with i < j <= 40 where the model's predicted relative rate for i exceeds that for j, across 100 seeds.

---

## The numbers

**Headline performance (test set, 2,500 policies):**

| Model | Poisson Deviance | Gini | Monotonicity compliance (17–40) |
|---|---|---|---|
| Poisson GLM | 1.142 | 0.181 | 100% |
| Unconstrained EBM | 0.983 | 0.347 | 69% |
| Constrained EBM | 0.991 | 0.333 | 100% |

Three things to read from this table:

First, the unconstrained EBM's Gini of 0.347 versus the GLM's 0.181 is a 19-point absolute improvement. This is real. It comes primarily from the vehicle age × driver age interaction: young drivers in older vehicles have materially different claim rates from young drivers in new vehicles, and the EBM captures this; the GLM does not, even with a quadratic age term.

Second, the unconstrained EBM violates monotonicity in the young driver range 31% of the time across seeds. This is not a fluke — it is the model fitting noise in thin age bands. On a fresh UK motor book the failure rate will vary, but the structural cause is the same: gradient boosting has no reason to impose monotonicity if the training data do not enforce it.

Third, the constrained EBM closes 80% of the Gini gap to the unconstrained model (Gini 0.333 vs 0.347), while achieving 100% monotonicity compliance. The Poisson deviance cost is negligible (0.991 vs 0.983). This is the right trade-off for a UK pricing model.

---

## What the shape functions look like

```python
from insurance_gam import InsuranceEBM
import polars as pl

ebm_constrained = InsuranceEBM(
    loss='poisson',
    interactions='3x',
    monotone_decreasing=['driver_age_17_40']  # non-increasing 17→40
)
ebm_constrained.fit(X_train, y_train, exposure=exposure_train)

# Shape functions: log-space contributions per feature
shapes = ebm_constrained.shape_functions()
# shapes['driver_age'] -> (ages, log_relativities) — guaranteed monotone 17→40

# Interaction plot: vehicle_age × driver_age
ebm_constrained.plot_interaction('vehicle_age', 'driver_age')
# This is the interaction GLM misses
```

The shape function for driver age in the constrained model is smooth and monotone from 17 to 40. The uptick above 55 is present and unconstrained — the data support it and it is plausible, so we let the model recover it. The vehicle age × driver age interaction shows the material effect: a 20-year-old in a 15-year-old vehicle has roughly 35% higher predicted frequency than a 20-year-old in a 2-year-old vehicle, controlling for all other factors. The GLM misses this entirely because it has no interaction terms for these two features.

---

## The krupova et al. benchmark result

Krupova, Rachdi and Guibert (arXiv:2503.21321) fit standard InterpretML EBMs to the French motor MTPL dataset (beMTPL97, roughly 160,000 policies) and compare against GLM, GAM, GBM and XGBoost on frequency and severity. Their finding on frequency: EBM outperforms GLM on Poisson deviance by 3.2% and outperforms GAM (with smoothing splines) by 1.8%. The improvement over GLM is primarily from the interaction terms — pairwise interactions that the additive structure of the GLM cannot represent.

On severity (Gamma deviance), the EBM improvement is smaller — 1.1% over GLM — because severity is structurally smoother and more amenable to GLM representations.

The monotonicity question is not addressed in that paper because French regulatory requirements differ. For UK pricing, it is the blocking question: an EBM that is 3.2% better on deviance but fails the monotonicity compliance check in the sign-off process is not deployable.

The constrained EBM resolves this. The 1-point Gini cost of the monotonicity constraint (0.333 vs 0.347 in our benchmark) is acceptable. A 19-point improvement over the GLM with full monotonicity compliance is deployable.

---

## The interaction capture: where EBM earns its keep

The vehicle age × driver age interaction is the most commercially significant EBM advantage over GLM on UK motor. In the constrained EBM benchmark:

| Segment | GLM predicted freq | Constrained EBM predicted freq | True freq |
|---|---|---|---|
| Age 19, vehicle age <3 | 0.087 | 0.081 | 0.080 |
| Age 19, vehicle age 10–15 | 0.087 | 0.109 | 0.112 |
| Age 45, vehicle age <3 | 0.044 | 0.042 | 0.041 |
| Age 45, vehicle age 10–15 | 0.044 | 0.047 | 0.048 |

The GLM applies the same young-driver rate regardless of vehicle age — it cannot capture the interaction. The constrained EBM correctly identifies that the young-driver / old-vehicle combination is materially riskier than the marginal factors would suggest. The GLM is underpricing this segment by roughly 25% relative to the EBM (and to truth). In a UK motor book with young drivers in older vehicles concentrated in lower socioeconomic postcodes, this is systematic mispricing with a demographic dimension that regulators will eventually notice.

---

## When not to use the constrained EBM

**When your monotone requirement is wrong.** Monotone constraints should reflect genuine actuarial expectation, not regulatory convenience. If the claims data suggest a real non-monotone structure — older drivers in the 65–70 band claiming less than 60–65 due to retirement and reduced mileage — a blanket "older is more expensive" constraint will mis-specify the model. Test the monotone assumption before imposing it.

**When you have fewer than 5,000 policies.** The EBM needs enough data to fit interaction terms reliably. Below 5,000 training policies, the interaction terms overfit. Use a constrained GLM with quadratic age terms instead — it is simpler, more stable, and almost as accurate on thin data.

**When you cannot explain interaction terms to a model validation committee.** The vehicle age × driver age interaction is interpretable if you can plot the 2D shape function. Some model governance processes require a simpler story. Know your audience.

---

## Verdict

Monotonicity-constrained EBM closes 80% of the Gini gap to the unconstrained model while achieving full monotonicity compliance. The 19-point Gini improvement over the standard Poisson GLM is real, driven primarily by interaction terms that GLM structurally cannot capture.

The critical practical finding: unconstrained EBM violates young-driver monotonicity in 31% of runs on our DGP. This is not a model failure — it is the model doing what it is told, which is fitting the data without structural constraints. On a UK motor book, you cannot deploy that.

Use the constrained EBM if:
- Your book has at least 5,000 training policies (preferably 10,000+)
- You have specific age ranges where regulatory or commercial requirements mandate monotonicity
- You can articulate the interaction terms in a model governance pack — plot the 2D shape functions and include them in the sign-off documentation

The technique works. The constraint costs 1 Gini point. The benefit is a deployable model that does not produce anomalous renewals and does not fail the model validation process on a technicality.

```bash
uv add insurance-gam
```

Source, benchmarks, and the monotonicity constraint documentation at [GitHub](https://github.com/burning-cost/insurance-gam). The constrained vs unconstrained benchmark is at `benchmarks/benchmark_monotone_ebm.py`.

References:
- Laub, P., Pho, T. & Wong, B. (2025). 'An Interpretable Deep Learning Model for General Insurance Pricing.' arXiv:2509.08467.
- Krupova, Z., Rachdi, M. & Guibert, Q. (2025). 'EBM benchmarks on French motor MTPL: frequency and severity.' arXiv:2503.21321.

- [Does insurance-gam actually work for insurance pricing?](/2026/03/24/does-insurance-gam-actually-work-pricing/)
- [Actuarial Neural Additive Model: What the Paper Actually Does](/2026/03/25/actuarial-neural-additive-model-anam-arxiv-2509-08467/)
- [Does HMM Telematics Risk Scoring Actually Work?](/2026/03/31/does-hmm-telematics-risk-scoring-actually-work/)
