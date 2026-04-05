---
layout: post
title: "GAMformer: A Genuinely Clever Idea That Cannot Help You Today"
date: 2026-04-03
categories: [techniques, interpretability]
tags: [GAM, EBM, interpretability, in-context-learning, transformer, GAMformer, InterpretML, insurance-gam, shape-functions, arXiv-2410.04560, pricing, TabPFN]
description: "GAMformer produces GAM shape functions in a single forward pass, no hyperparameter search, sub-second inference. For insurance pricing today: three hard blockers — max 500 rows, classification only, no released weights. EBM already solves the problem. Watch list, not build list."
author: burning-cost
---

Fitting an Explainable Boosting Machine (EBM) on a personal lines dataset is painfully slow. On a 500,000-policy motor book with 30 features, a full EBM fit with default settings takes 20-40 minutes. You pay that cost every time you want to explore whether a new engineered feature is doing anything useful, every time you want to see whether `vehicle_age` has a step-change or a smooth gradient, every time you iterate on a rating structure before committing to a refactor.

GAMformer (arXiv:2410.04560, Mueller et al., NeurIPS 2024 TRL Workshop) aims to solve that problem. The idea is genuinely novel: a 50.8M-parameter transformer that reads your training data as context and outputs GAM shape functions directly, in a single forward pass. No iterative fitting. No hyperparameter search. Plausibly sub-second inference.

It does not work for insurance pricing today. Not because the idea is wrong, but because the current implementation has three blockers that are not edge cases — they are central architectural constraints. This post explains what GAMformer actually does, why the core idea is worth watching, and what would need to change before it belongs in your workflow.

---

## What GAMformer actually does

Standard GAM fitting — EBM, pyGAM, mgcv — is iterative. Gradient boosting or penalised spline fitting, many passes over the data, convergence checks. That is where the time goes.

GAMformer reframes this as in-context learning. The transformer takes your entire training set as input:

```
Context: [(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)]
Query:   [x_test]
Output:  shape function values f_i(x_i) for each feature i, plus prediction
```

The prediction is the standard GAM structure: a sum of per-feature shape functions plus a link function. The shape functions themselves are represented as 64-bin histograms — non-parametric and directly interpretable.

The architectural move that makes this work is a dual self-attention design: attention is applied first over features (within each data point), then over data points (within each feature). This lets the model learn the joint signal of feature interactions and cross-observation patterns simultaneously.

The transformer was pre-trained on over 100 million synthetic datasets, generated from Structural Causal Models (random DAGs with 96% weight) and Gaussian Processes (4%). Training ran for 25 days on a single A100. At inference time, the model has already learned what GAM shapes look like across a vast prior of possible data-generating processes. It does not iterate — it reads the data once and produces shapes directly.

The shapes the paper reports are visually smoother than EBM outputs. That smoothness comes from the prior, not the data. Whether you find that attractive depends on whether you trust the prior more than your data, and for insurance pricing, you should not.

---

## Three hard blockers

### 1. Maximum 500 rows

The transformer was trained on datasets of at most 500 samples. The paper states directly:

> "difficulty in improving predictions when presented with datasets that exceed twice the size of the data it saw during training"

This is not a tuning parameter or a default you can override. It is baked into the positional representations the model learned during pre-training. Performance degrades when presented with more than roughly 1,000 rows, because the model has never learned the positional attention patterns that correspond to larger datasets.

UK personal lines portfolios run to 200,000–2,000,000 policies. You could sample 500 rows and run GAMformer on the sample. That is methodologically unsound — 500 observations cannot establish the marginal effect of `vehicle_group` with 50 rating levels — and it would not survive actuarial peer review or FCA governance. Bootstrapping over many 500-row samples defeats the speed advantage and adds variance without fixing the representativeness problem.

On the benchmarks in the paper, the accuracy gap relative to EBM already widens as datasets grow: the MIMIC-III result shows GAMformer 6.7 percentage points behind EBM on AUC. The row limit is the most likely explanation.

### 2. Classification only

The training objective throughout is cross-entropy. The pre-trained model knows how to produce shapes for binary classification targets. It does not know about Poisson distributions, Gamma distributions, log-links, or overdispersion.

Insurance frequency modelling requires Poisson or negative binomial regression. Severity modelling requires Gamma or lognormal. Commercial lines pricing uses Tweedie. A GAMformer that outputs logit-scale shapes from a cross-entropy-trained prior is not a frequency model or a severity model — it is a classifier.

Adding Poisson/Gamma support would require retraining the full transformer with deviance-based training objectives on synthetic count and positive-continuous data. That is approximately 25 A100-days — a substantial research project, not a configuration flag. The paper gives no indication this is planned.

For comparison: EBM with Poisson loss is already in InterpretML and has been for years. It works on millions of rows. This is not a gap you need GAMformer to fill.

### 3. No released weights

The code lives in `microsoft/ticl`, a repository that also contains MotherNet and TabFlex. The GAMformer section of the README is explicitly marked **WIP**. There is no code, no documentation, and no model checkpoint. There is no HuggingFace model page. There is no pip package.

Without released weights, you cannot run GAMformer at all. The alternative — self-training — requires implementing the synthetic data generation pipeline, running 25 days of A100 compute, and validating the result against the paper's benchmarks. That is not a realistic option for a pricing team.

The v2 revision (February 2026) is 16% larger by file size than v1, which suggests expanded content — possibly regression benchmarks or extended results — but the HTML version was returning 404 as of April 2026, so we cannot confirm what changed.

---

## The accuracy gap on what it does support

Even setting aside the three blockers, GAMformer is consistently behind EBM on the benchmarks it was actually evaluated on. Across five binary classification datasets:

| Dataset | GAMformer AUC | EBM AUC | Gap |
|---------|---------------|---------|-----|
| Churn | 81.7% | 83.6% | 1.9pp |
| Support2 | 80.8% | 82.4% | 1.5pp |
| Adult | 90.1% | 93.1% | 3.0pp |
| MIMIC-II | 82.2% | 85.2% | 2.9pp |
| MIMIC-III | 74.4% | 81.1% | 6.7pp |

The smoother shapes in the paper are visually appealing, but smoother with lower AUC is not a better model — it is an underfit one. For governance purposes, a shape function that reflects the actual data relationship (even if noisier) is preferable to one that reflects the prior's smoothness assumption.

---

## What already works: EBM via insurance-gam

The problem GAMformer is trying to solve — slow GAM fitting, inaccessible shape functions — already has a practical solution.

The `insurance-gam` library provides an `EBMInsuranceWrapper` built on InterpretML's EBM. You get:

- Poisson and Gamma loss functions for frequency and severity
- sklearn-compatible API: `fit()`, `predict()`; global explanations via `wrapper.model_.explain_global()` on the underlying InterpretML EBM object
- Shape function plots via InterpretML's visualisation
- Feature importance from the boosting weights
- Works on full UK personal lines portfolios (tested to 1M+ rows)

For quick exploratory shape checks — the specific use case GAMformer targets — the right approach is to restrict EBM fitting with `max_rounds` and `min_samples_leaf`. A fast approximate EBM on 50,000 stratified rows fits in under two minutes and gives shape functions that are representative of the full portfolio. Alternatively, pyGAM fits in under five seconds on 500 rows, which is faster than whatever GAMformer's forward pass would actually be on the hardware most pricing teams have access to.

---

## Why the idea is still worth watching

The core concept — using in-context learning to produce GAM shapes from a pre-trained prior, without any iterative fitting — is architecturally novel. This is not a repackaged EBM or a minor modification of existing GAM software. It is a new mechanism for producing interpretable shape functions, and for certain applications the sub-second inference speed would be genuinely valuable.

The proof of concept is plausible. On small classification tasks, GAMformer demonstrates that the approach works. The shapes are interpretable. The mechanism generalises from synthetic to real-world data. The NeurIPS 2024 TRL Workshop acceptance confirms that the ideas are sound enough to interest the ML interpretability community.

What it needs to become useful for insurance is not small changes — it needs Poisson/Gamma regression, row scaling by a factor of 1,000, and released weights. Those are substantial asks. But the direction is right, and if a future version delivered all three, the speed advantage over EBM would be real and the hyperparameter-free fitting would be genuinely useful for pricing exploratory work.

---

## Watch triggers

Check `microsoft/ticl` quarterly. Upgrade to BLOG or BUILD when:

1. **Weights released.** GAMformer README moves from WIP to documented, with a downloadable checkpoint.
2. **Regression families added.** Any evidence of Poisson or Gamma training in v3+ or a new repository.
3. **Row limit extended.** Paper or code showing stable performance on datasets above 10K rows, implying architectural changes to the attention mechanism (linear attention, chunked attention, or similar).
4. **pip package.** A standalone `gamformer` package on PyPI with a documented API.

None of these conditions are currently met.

---

## The bottom line

GAMformer is a research paper about an interesting idea, not a tool for UK insurance pricing in 2026. The three blockers — 500-row limit, classification only, no released weights — are not implementation oversights. They reflect the scope of what was built: a proof of concept for in-context GAM learning on small binary classification tasks.

The practical alternative for every use case we can identify is already available. Fast shape exploration: `max_rounds`-restricted EBM or pyGAM. Production frequency modelling: EBM with Poisson loss via `insurance-gam`. Production severity modelling: EBM with Gamma loss or a distributional GBM. None of these require 25 days of A100 compute or waiting for Microsoft Research to release checkpoints.

The paper is Mueller, Siems, Nori, Salinas, Zela, Caruana and Hutter, "GAMformer: Bridging Tabular Foundation Models and Interpretable Machine Learning," arXiv:2410.04560v2, February 2026. Verdict: WATCH (9/20). Not because the work is weak, but because it is not finished yet.
