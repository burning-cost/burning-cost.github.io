---
layout: post
title: "Tabular Foundation Models for Insurance Pricing — Do They Work?"
date: 2026-03-28
categories: [techniques, research]
tags: [foundation-models, tabpfn, tabicl, tab-trm, xgboost, catboost, tabred, benchmarks, poisson, exposure-offset, frequency-severity, pricing, actuarial, uk-motor, in-context-learning]
description: "An honest assessment of where tabular foundation models stand in March 2026 — what the benchmarks actually show, what's missing for insurance pricing, and which models are worth watching. The verdict: XGBoost/CatBoost hold. Tab-TRM is the one to watch."
---

The question actuaries are now being asked — in model review meetings, at conference panels, and in vendor pitches — is whether the new generation of tabular foundation models has made gradient-boosted trees obsolete for insurance pricing.

The honest answer, as of March 2026: no. But the question is no longer absurd, and some of the models being built specifically for insurance are genuinely different from the general-purpose tools that have been failing this test for the past five years.

Here is a practitioner's assessment of the landscape, grounded in the benchmarks and the specific requirements of insurance frequency/severity modelling.

---

## What tabular foundation models actually are

There are two distinct paradigms worth separating, because they have very different practical implications.

**In-context learning (ICL) models** — TabPFN v2 ([arXiv:2501.02945](https://arxiv.org/abs/2501.02945), *Nature* 637:319–326, Jan 2025), TabPFN 2.5 ([arXiv:2511.08667](https://arxiv.org/abs/2511.08667), Nov 2025), and TabICL v2 ([arXiv:2502.05564](https://arxiv.org/abs/2502.05564), ICML 2025) — do not train on your data at all. They are transformers pre-trained on millions of synthetic tabular datasets. At inference time, your training data is fed as a prompt (context) and predictions come out of a forward pass. No gradient steps. No hyperparameter search. For a 1,000-row dataset, the full cycle takes a few seconds on a CPU.

This is not fine-tuning and not transfer learning. It is Bayesian inference over a very broad prior about what tabular data-generating processes look like — learned by pre-training rather than expressed analytically.

**Fine-tuned or insurance-specific models** take a different route. Tab-TRM ([arXiv:2601.07675](https://arxiv.org/abs/2601.07675), Padayachy, Richman, Wüthrich, Jan 2026) is pre-trained and then fine-tuned on insurance data, with the architecture explicitly designed around Poisson deviance loss and exposure offsets. XTab and UniPredict attempted cross-dataset pre-training on real tabular data, with mixed results.

Why should actuaries care about either? The ICL models promise zero hyperparameter tuning and genuine skill on small datasets. The insurance-specific models promise to solve the actuarial loss function problem. Neither promise is fully delivered yet — but the gap is closing.

We covered the thin-data use case for TabPFN v2 specifically in an [earlier post]({% post_url 2026-03-25-tabular-foundation-models-thin-segment-insurance %}). This post covers the full landscape.

---

## What the benchmarks actually show

The empirical picture is more nuanced than either the hype or the backlash suggests.

**On standard ML benchmarks with random splits**, TabPFN 2.5 and TabICL v2 are competitive with or better than XGBoost/CatBoost on datasets up to roughly 10,000 rows. TabICL v2 is state-of-the-art on the TALENT benchmark (300+ datasets) and TabArena as of March 2026. Real-TabPFN-2.5 (fine-tuned on real data) matches AutoGluon 1.4 extreme — a four-hour ensemble — on TabArena-lite. This is not nothing.

**But TabReD ([arXiv:2406.19380](https://arxiv.org/abs/2406.19380), ICLR 2025 Spotlight) is the reality check.** Eight industrial datasets drawn from Kaggle and production systems, with temporal train/test splits rather than random cross-validation. The finding is stark: model rankings change materially when you use temporal splits. Complex deep learning models — attention-heavy architectures, transformers — that look strong on random splits do not transfer those gains to temporal evaluation. Simple MLPs and GBDTs perform best.

The paper puts it plainly: "more complex DL methods turn out to be less effective in the new setting."

This is directly relevant to insurance. A model trained on 2024 policies and tested on 2025 renewals faces exactly the kind of temporal distribution shift TabReD tests for. Claims inflation, legislative changes, postcode gentrification, weather — insurance data is inherently temporal. A benchmark result on random k-fold cross-validation is insufficient evidence for production deployment. We require temporal validation, and under temporal validation, the tree models hold their ground.

The TabArena/TALENT benchmarks still have value — they tell you what is possible on structured tabular problems. But the TabReD result means you should be sceptical of any vendor or paper that benchmarks a tabular DL method on random splits and declares it ready for production pricing.

---

## The insurance-specific problems

Even setting aside temporal evaluation, the general-purpose foundation models have gaps that are not minor inconveniences — they are fundamental blockers for core insurance pricing tasks.

**Exposure offsets.** A standard Poisson frequency GLM fits:

```
log(E[claims]) = log(exposure) + Xβ
```

The log(exposure) term enters as a fixed offset, not a feature. This is not a technicality. It is what makes the model a rate model rather than a count model. A policy with six months of exposure should predict half the claims of an equivalent twelve-month policy, holding everything else equal. Without this, your frequency model is wrong by construction for any cohort with mixed policy terms — mid-term cancellations, new joiners, inception bias.

Neither TabPFN nor TabICL supports exposure offsets natively. The workaround — passing log(claims/exposure) as the target — changes the learning objective and is miscalibrated for partial-year policies. This is a hard blocker for frequency modelling. XGBoost and CatBoost both support exposure offsets directly via the `base_margin` or offset parameter. GLMs support it by definition.

**Actuarial loss functions.** Insurance pricing optimises Poisson deviance (frequency), Gamma deviance (severity), or Tweedie deviance (pure premium). These are not cosmetically different from mean squared error. The choice of loss determines what the model is fitting, affects outlier robustness, and matters for calibration under skewed claim distributions.

TabPFN v2 and TabICL treat regression as Gaussian-equivalent under the hood. There is no native Tweedie, no native Poisson, no native Gamma. XGBoost supports all three with `objective='tweedie'`, `'poisson'`, `'gamma'`. LightGBM and CatBoost similarly.

**High-cardinality categorical features.** UK motor pricing involves around 1.7 million postcode sectors and roughly 60,000 vehicle make-model combinations. CatBoost's ordered target encoding handles this robustly and is the industry default. TabPFN degrades on high-dimensional data — performance falls below logistic regression when features exceed 2,000. High-cardinality encoding of postcodes alone would breach this limit. TabICL handles more features but its behaviour on insurance-grade categorical cardinality has not been documented.

**Dataset scale.** TabPFN v2 has a hard limit of 10,000 rows — it is out of scope for any main UK motor book. TabPFN 2.5 extends this to 50,000 rows. TabICL scales to 500K rows via CPU offloading, which covers most UK personal lines books. But neither has been validated on insurance data with temporal splits at this scale. XGBoost and CatBoost have no hard row limits and have been production-validated on hundreds of millions of policies.

These are not gaps that a bright engineer can close in a sprint. Exposure offset support requires rethinking the training objective at architecture level. Actuarial loss functions require pre-training the foundation model with those losses, not adding them at fine-tune time. The categorical cardinality problem may be genuinely unsolvable within the current ICL paradigm — you cannot put a one-million-category embedding in a prompt.

---

## Tab-TRM: the one to watch

Tab-TRM is different from the general-purpose models and deserves separate treatment.

Padayachy, Richman, and Wüthrich — who between them have written much of the modern actuarial deep learning literature — published this in January 2026. The architecture is a Tiny Recursive Model: two learnable latent tokens (an answer embedding and a reasoning state) iterated recursively. At 14,820 parameters, it achieves equivalent depth to a roughly 4,050-layer feedforward network through parameter reuse. The exposure offset is explicitly incorporated into the Poisson likelihood — not a workaround, but a first-class architectural feature. Loss is Poisson deviance. This is, as far as we know, the only mainstream tabular neural network designed from the ground up around insurance actuarial requirements.

The benchmark results on the French MTPL dataset (Dutang et al. 2024) are the most relevant insurance-specific numbers available:

| Model | OOS Poisson deviance (×10⁻²) |
|---|---|
| Null model | 25.445 |
| Poisson GLM | 24.102 |
| Credibility Transformer ensemble | 23.711 |
| Tree-like PIN ensemble | 23.667 |
| Tab-TRM (10-run ensemble) | **23.589** |

Tab-TRM is best. The GLM-to-Tab-TRM improvement is approximately 2.1%. On a one-million-policy book, that is material. The comparison matters because this is the right benchmark — Poisson deviance, exposure-adjusted, against real actuarial competitors — not a generic ML suite.

There is a further interpretability point: the internal token dynamics in Tab-TRM are well-approximated by linear maps (R² > 0.8 on hidden-state updates). The model is quasi-linear by construction, which is relevant for UK regulatory sign-off under PRA SS1/24's "generally accepted market practice" standard.

The limitations are honest ones. Tab-TRM has been validated on one dataset: the French MTPL portfolio. There is no UK validation, no home insurance validation, no commercial lines validation. It is not pip-installable — the GitHub repository ([SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)) contains research code only, with no public model weights. There is no severity (Gamma) model in the published results — frequency only. And the French MTPL benchmark uses random splits, not temporal splits; we do not know how Tab-TRM behaves under TabReD-style evaluation on insurance data.

These are not permanent limitations. They are the expected state of a well-designed first paper. But they mean that Tab-TRM is not a production option today, and anyone who tells you otherwise has not read the paper.

---

## What to actually do

For most pricing teams in March 2026, the practical answer is straightforward:

**For production frequency/severity modelling (50K–5M policies):** XGBoost or CatBoost with Poisson/Gamma/Tweedie loss and proper exposure offsets. This is not a counsel of conservatism — it is what the evidence, evaluated under realistic temporal conditions, supports. The GBDTs are not being embarrassed by TabReD; they are being vindicated. The appropriate comparison model for a new tool is not "GLM" but "tuned CatBoost with temporal validation," and almost nothing beats that outside the thin-data regime.

**For thin segments (fewer than ~1,000 policies):** TabPFN v2 or TabICL v2 are worth experimenting with for classification tasks (fraud, lapse, churn) and potentially for relative risk ranking in new product pricing. The absence of exposure offset support means these are not drop-in replacements for frequency models, but they can complement a credibility framework. We have covered this use case in detail [previously]({% post_url 2026-03-25-tabular-foundation-models-thin-segment-insurance %}).

**For Tab-TRM:** add it to your watching brief. If the Tab-TRM paper is followed by pip packaging, public weights, independent replication on a second dataset, and a temporal validation, then the calculus changes — this is the first neural model that has been designed correctly for insurance frequency modelling, and the benchmark result is strong. The right response now is to read the paper (arXiv:2601.07675), understand the architecture, and be ready to run your own validation when the tooling matures.

**For model governance:** require any benchmark of a tabular foundation model to include temporal splits before it reaches a model risk committee. Results on random k-fold cross-validation are not sufficient for insurance pricing decisions. This is the main practical lesson from TabReD, and it is cheap to implement as a governance standard.

The hype cycle for tabular deep learning has been running for several years. The models have improved significantly — TabICL v2 and TabPFN 2.5 are genuinely better than their predecessors. But "improved" is not the same as "better than CatBoost for insurance pricing in production." The gap that remains is not primarily about accuracy on standard benchmarks. It is about the insurance-specific requirements — exposure offsets, actuarial loss functions, temporal robustness — that the general-purpose models were not built to meet, and that Tab-TRM was built to meet but has not yet demonstrated at scale.

Watch the Tab-TRM GitHub. Track the TabReD follow-on work ([arXiv:2502.20260](https://arxiv.org/abs/2502.20260), ICML 2025). Keep your CatBoost pipelines in production.
