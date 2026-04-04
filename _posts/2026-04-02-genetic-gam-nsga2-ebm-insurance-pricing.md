---
layout: post
title: "Genetic GAMs: An Interesting Idea That EBM Already Solved"
date: 2026-04-02
categories: [machine-learning, research]
tags: [gam, ebm, interpretable-models, nsga-ii, genetic-algorithm, insurance-gam, interactions, hyperparameter-search, actuarial, arXiv-2602.15877]
description: "Shankar & Cohen automate GAM structure search using NSGA-II evolutionary algorithms. The idea is legitimate; the problem is that EBM already does this better for insurance pricing, without swapping one set of hyperparameters for another."
author: burning-cost
---

Configuring a GAM by hand is tedious. How many knots? Where? Which pairs of variables should interact? What smoothing penalty? Actuaries developing motor frequency models iterate through this manually or via cross-validation grids, and the answer changes with the data. Shankar & Cohen (arXiv:2602.15877, NAFIPS 2026) propose automating the whole search using NSGA-II — a multi-objective genetic algorithm that jointly minimises prediction error and a complexity penalty capturing sparsity, smoothness, and uncertainty.

The idea is legitimate. The execution, for insurance pricing, is not convincing.

---

## What the paper does

Each NSGA-II "chromosome" encodes a complete GAM specification: knot placement and count, interaction terms, smoothing penalties, and basis function type (thin-plate splines, cubic regression splines). The algorithm evaluates fitness by fitting the encoded GAM on training data and measuring test RMSE plus a complexity score. After many generations, it produces a Pareto front of accuracy-versus-complexity trade-offs rather than a single winner.

The Pareto front is actually the most genuinely useful part of the contribution. Rather than committing to one configuration, you get a family of models at different accuracy-complexity points, and you can reason about how much interpretability you are paying for each unit of predictive gain. That is a useful diagnostic, even if the optimisation itself is slow.

Results are reported on the California Housing dataset. NSGA-II GAMs outperform baseline LinearGAMs and claim narrower confidence intervals than unoptimised specifications.

That is where our enthusiasm runs out.

---

## Why EBM has already solved the harder version of this

The paper positions itself as automating what practitioners do manually — GAM structure search. But [EBM (InterpretML)](https://interpret.ml/docs/ebm.html) solved this problem via gradient boosting, not evolutionary search, and it solved a harder version of it.

EBM does not require manual knot placement at all. The boosting procedure iteratively fits one-dimensional shape functions, implicitly determining where curvature is needed by following the gradient. The FAST algorithm detects candidate interaction pairs by measuring the residual signal between variables after univariate fitting. You do not specify knots; you specify the number of boosting rounds and the maximum number of interaction terms. The result is a model with the same additive + interaction structure as a GAM, fully interpretable, without a chromosome in sight.

Our [insurance-gam library](/insurance-gam/) wraps EBM with `EBMInsuranceWrapper`, adding Poisson and Gamma objectives with exposure offsets, monotonicity constraints for regulatory explainability, and calibration diagnostics. For a UK motor pricing team, this is the baseline that any automated GAM configuration method must beat. The genetic paper does not attempt this comparison seriously.

---

## The meta-problem: swapping one set of knobs for another

There is a deeper issue with evolutionary hyperparameter search that the paper acknowledges but does not fully reckon with: every NSGA-II run requires its own configuration. Population size, number of generations, crossover rate, mutation rate — these are new degrees of freedom that have replaced the GAM's knot and penalty parameters.

Cross-validating over a grid of smoothing penalties for a 10-variable GAM is annoying but tractable. Running a genetic algorithm that requires fitting potentially thousands of GAM candidates per generation, on an insurance dataset of several million policy-years, is not. The paper notes computational cost as a limitation and offers no solution. On a mid-size insurer's dataset, this approach would take hours to days per model development cycle.

EBM trains in minutes on the same data. The tradeoff is not close.

---

## What is missing for the insurance case

The paper uses RMSE as its sole objective. Insurance frequency modelling requires Poisson or negative binomial objectives; severity modelling requires Gamma. RMSE is not appropriate for count data with exposure offsets and heavy right tails. Without actuarially appropriate loss functions, any comparison to insurance pricing methods is category error.

There is also a single dataset. No French MTPL, no Allstate claims, no Lloyd's open market data. One California Housing experiment does not establish that the approach transfers to insurance-specific data-generating processes.

The venue matters here too. NAFIPS 2026 is a fuzzy systems workshop. This is an early-stage idea presented to a sympathetic community, not a result that has been stress-tested by a statistical learning audience.

---

## What would change our assessment

If the genetic search were demonstrated to find interaction terms that EBM misses — particularly three-way interactions that EBM's pairwise FAST algorithm cannot detect — and if it did so with a Poisson objective on a real insurance dataset, that would be a meaningful contribution. The Pareto front concept applied to actuarial model selection (how much Poisson deviance do you pay per interaction term removed?) is genuinely interesting.

We do not think evolutionary GAM search will beat EBM on speed or accuracy in insurance use cases. But if it turns out that genetic search reliably identifies specific interaction pairs that gradient boosting misses in sparse, high-dimensional claim datasets, the idea deserves more serious treatment.

---

## What this means for our stack

Nothing changes for now. [`insurance-gam`](/insurance-gam/) already has:

- `EBMInsuranceWrapper` — automated shape functions with Poisson/Gamma objectives and exposure offsets
- `ActuarialNAM` — neural additive model variant with the same actuarial loss functions
- `PIN` subpackage — Pairwise Interaction Networks for targeted two-way interaction detection

The genetic GAM is attempting what PIN does (interaction selection) and what EBM does (shape complexity optimisation), but with a slower search and no actuarial loss functions.

Worth knowing about. We will revisit if code appears and a Poisson comparison against EBM on a standard insurance dataset is published.

**Paper:** [arXiv:2602.15877](https://arxiv.org/abs/2602.15877)
