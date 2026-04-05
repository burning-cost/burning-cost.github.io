---
layout: post
title: "Why Stacking Fairness, Calibration, and Drift Corrections Breaks Your Pricing Model"
date: 2026-04-05
author: Burning Cost
categories: [fairness, calibration, monitoring, model-governance]
tags: [fairness-constraints, calibration, drift-detection, consumer-duty, model-risk, tweedie-glm, demographic-parity, PSI, isotonic-regression, joint-optimisation, arXiv-2603.06733, insurance-fairness, insurance-monitoring, insurance-distributional-glm]
description: "Applying fairness constraints, calibration corrections, and drift monitoring as sequential post-hoc steps is how most UK pricing teams work. It is also architecturally broken. Each correction partially undoes the previous one. This is an honest account of the problem — including the fact that our own libraries do exactly this."
math: true
---

Most UK pricing teams applying model governance corrections follow roughly the same pipeline: train a GLM or GBM, apply fairness constraints to satisfy Consumer Duty, run isotonic recalibration to tighten the expected calibration error, then set up PSI-triggered monitoring to catch drift and trigger a refit. Repeat annually, or when something breaks.

This pipeline is common because its components are individually well-motivated. Fairness constraints are not optional under the FCA's Consumer Duty (PS22/9) and the Equality Act 2010. Calibration matters for reserving and pricing adequacy. Drift detection — flagged in PRA consultation CP6/24 on insurance model risk — is how you know when to refit. Each step corresponds to a genuine regulatory obligation or good practice standard.

The problem is that these steps interact. Applied sequentially as post-hoc corrections, they do not compose cleanly. Each adjustment changes the distribution the next one operates on. The result is a model that has been corrected three times but is not correctly corrected in any of them.

We are not exempt from this criticism. Our own libraries — `insurance-fairness`, `insurance-monitoring`, `insurance-distributional-glm` — treat these as separate concerns. That was the right engineering decision for getting useful tools into the hands of pricing teams. It is also a known limitation that we want to be honest about.

---

## What the standard pipeline looks like

Start with a Tweedie GLM fit on UK motor data: three years of policy and claims history, log link, variance power $$p = 1.5$$, covering both frequency and severity in a single model. Standard stuff.

Step one: apply a demographic parity constraint on gender. The FCA's Consumer Duty fair value expectations and the December 2025 Research Note on motor insurance pricing and local area ethnicity mean you need to check that postcode (or any other rating factor correlated with a protected characteristic) is not acting as a proxy for a protected characteristic. You run the audit, find that three postcode bands produce a demographic parity gap above your internal threshold, and apply a Wasserstein transport correction to bring the group-conditional prediction distributions closer together. The DP gap moves from 0.061 to 0.029. You document it. Sign-off obtained.

Step two: calibrate. The fairness correction has shifted the prediction distribution for the affected postcode groups. Their scores have been redistributed — that is what the transport correction does. But isotonic recalibration was fitted on the original model's outputs, not on the fairness-corrected outputs. You refit the isotonic mapping on the corrected scores. The Expected Calibration Error drops from 0.041 to 0.019. Good.

Step three: deploy. Set up PSI monitoring on the feature distribution. Six months later, PSI exceeds 0.2 on vehicle age — a threshold that in our experience triggers a model review — and you trigger a partial refit on vehicle age using the new data. The refit is on the base Tweedie GLM, which is then pushed back through the fairness correction and recalibration steps.

You now have a model that has been through three independent correction passes. The question is whether those three corrections have achieved what you think they have.

---

## Why they do not compose

The failure mode has a precise structure. Each correction is fitted to the output of the model it receives. When you change the model — through a downstream correction or a refit — the upstream correction is no longer fitted to the right inputs.

**Fairness correction → calibration.** The Wasserstein transport step redistributes scores across demographic groups to achieve distributional balance. It does not preserve the marginal prediction distribution — that is not its objective. Isotonic recalibration fitted on the pre-correction scores is now mapping the wrong input space. The bins that isotonic regression cut through the original score distribution no longer correspond to the same risks. The ECE you measured was for a model that no longer exists.

To see this concretely: suppose the transport correction moves the 75th percentile score for Group A from 0.12 to 0.09. An isotonic bin that was calibrated to map [0.08, 0.13] → 0.105 is now receiving scores that used to sit in [0.06, 0.10]. The risks at that score point have changed. The calibration mapping is wrong at exactly the point where it matters most — the high-score segment where pricing decisions are concentrated.

**Calibration → fairness.** Now run it in reverse. If you recalibrate before applying the fairness correction, the calibration step implicitly sets a marginal score distribution. The fairness correction then transports scores to achieve group balance. But the transport distance depends on the starting distributions of each group. If Group A and Group B have been recalibrated to different distributional shapes — which is common when the groups have different claim frequency profiles — the transport correction is balancing distributions that have already been independently reshaped. The resulting DP gap will be different from what you would have achieved without the prior calibration step.

Neither order is obviously correct. The corrections are not commutative.

**Drift correction → fairness and calibration.** A PSI-triggered refit is where the composition problem becomes acute. When you detect feature drift and refit the base model on updated data, you get a new model with a new output distribution. You then push it through the existing fairness correction and calibration pipeline.

But the existing fairness correction was fitted on the old model's outputs. The transport maps were computed for the old score distribution. Applied to the new model's outputs, they are transporting from the wrong starting point. The DP gap you measured on the old model does not carry over. You have, in effect, applied corrections that were designed for a model you have now replaced.

If the refit was triggered by drift in a variable correlated with a protected characteristic — vehicle age and driver age are correlated with gender in UK motor data — then the new model's relationship between that variable and the protected characteristic may have shifted. The fairness correction fitted on the old model's outputs may now understate or overstate the needed correction. You could be making the model less fair while believing you have maintained it.

---

## The CCI paper and what it attempts

Nayak's Calibrated Credit Intelligence framework (arXiv:2603.06733, March 2026) is the closest published attempt we have seen to addressing this directly. The core architectural decision is to treat calibration, fairness, and temporal stability as joint training objectives rather than sequential post-hoc steps.

In CCI, a Bayesian neural network produces calibrated uncertainty estimates from the base scorer. A GBM layer is then trained with explicit fairness constraints — demographic parity and equal opportunity — baked into the loss function during gradient boosting, not applied after. The temporal component uses domain-adversarial training to make the features robust to distributional shift rather than flagging shift and refitting after the fact.

The paper reports a demographic parity gap of 0.046 and an Expected Calibration Error of 0.015 on the Home Credit Risk Model Stability benchmark. The ECE figure is particularly striking — getting there without post-hoc calibration simplifies the composition problem considerably.

There are reasons to treat this carefully. It is a single-author preprint from March 2026, from a credit risk domain. It has not been peer-reviewed. The benchmark it uses — the Home Credit Risk Stability competition dataset — is one dataset with one train/test structure. The model architecture is substantially more complex than a Tweedie GLM, and UK motor insurance pricing is not credit scoring. The gap between an AUC-ROC benchmark on a Kaggle-structured dataset and a regulatory-grade pricing model in production is wide.

But the architectural insight is sound. Joint optimisation avoids the composition problem precisely because there is nothing to compose — the constraints are enforced simultaneously during training, on the same gradient steps, with full visibility into how they interact.

---

## What a joint approach for insurance would require

The translation from CCI's architecture to a UK insurance pricing context is not trivial. The obstacles are specific.

**Tweedie loss and fairness constraints.** UK motor pricing models commonly use Tweedie GLMs for pure premium modelling. The standard fairness constraint literature — demographic parity, equal opportunity, counterfactual fairness — is developed almost entirely for classification. Extending demographic parity to a Tweedie distributional target requires specifying what parity means over a right-skewed, zero-inflated continuous distribution. The Wasserstein barycenter approach (as implemented in EquiPy and our own `insurance-fairness`) handles this by working in the distributional sense, but its integration into a training loop is non-trivial.

**Drift robustness during training vs at inference.** Domain-adversarial training — the mechanism CCI uses for temporal stability — requires you to specify the domains (time periods, geographic regions, underwriting year cohorts) at training time. In a production pricing model, the drift that triggers a refit is often not foreseeable at training time. You do not know in advance that vehicle age will drift in the way it did. Post-hoc PSI monitoring catches this. A jointly trained model would need to be retrained with the new domain anyway, at which point you are back to the composition problem on the next cycle.

**Regulatory auditability.** A constrained optimisation with multiple objectives — calibration loss, fairness penalty, distributional stability term — is harder to audit than a sequentially corrected model. With the sequential approach, each step can be documented independently: here is the fairness audit, here is the calibration report, here is the drift monitoring summary. With joint optimisation, the trade-offs between objectives are implicit in the loss weights. Explaining to a PRA model risk reviewer why the model accepted a 0.008 increase in ECE to achieve a 0.012 improvement in DP gap requires articulating those weights explicitly and defending them. That is not impossible, but it is new territory for most actuarial governance frameworks.

**Model complexity and stability.** A Bayesian neural network with GBM on top is a harder model to maintain than a Tweedie GLM. The pricing teams we work with are not uniformly equipped for Bayesian inference. Asking them to move from a GLM with post-hoc corrections to a constrained neural-GBM stack is a significant ask.

None of these are arguments against joint optimisation in principle. They are arguments for being realistic about the gap between the architectural insight and the production reality.

---

## Where this leaves UK pricing teams

The practical answer, for now, is: apply the corrections sequentially, but do it in the right order, refit the downstream steps whenever an upstream step changes, and document the interactions explicitly.

If you apply a fairness correction and then recalibrate, the calibration report in your model governance documentation must be for the fairness-corrected model, not the base model. If a drift-triggered refit changes the base model, the fairness audit and calibration must both be rerun — on the new model, in that order. These are not obvious to all governance frameworks, which often treat the fairness audit and the calibration report as one-time artefacts rather than things that are contingent on the model version they were run against.

More fundamentally, any monitoring regime that watches for calibration drift without also watching for fairness drift is incomplete. A PSI-triggered refit that fixes calibration may simultaneously move the DP gap. You need to know. `insurance-monitoring` and `insurance-fairness` can both be run on the post-refit model; the question is whether your governance process requires that, and whether it requires both to pass before the refitted model is deployed.

The composition problem does not go away by being careful about ordering and documentation. It is reduced to a manageable form, but the underlying issue — that these corrections were designed independently and do not know about each other — remains.

Joint optimisation during training is the right long-term answer. The CCI paper points in the right direction, even if the route from a credit risk preprint to a production UK motor GLM is not short. We expect the methods to mature over the next two to three years as the constrained learning literature develops actuarial-specific results and as the regulatory frameworks for explainability become clearer about what joint optimisation requires by way of documentation.

In the meantime, we are building towards better composition between our libraries — explicit APIs for running the full correction sequence in the right order, and monitoring that watches fairness and calibration jointly rather than separately. That work is ongoing and not finished. We are not going to pretend otherwise.

---

*The libraries referenced — `insurance-fairness`, `insurance-monitoring`, `insurance-distributional-glm` — are open source and available on PyPI. The CCI paper is arXiv:2603.06733. The fairness correction approach EquiPy uses is arXiv:2503.09866 (Fernandes Machado, Grondin, Ratz, Charpentier, Hu). FCA PS22/9 and PRIN 2A (Consumer Duty) cover fair value and proxy discrimination expectations; FCA EP25/2 (July 2025) evaluates the GIPP price-walking remedies; PRA CP6/24 covers insurance model risk management.*
