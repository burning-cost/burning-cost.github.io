---
layout: post
title: "Quantum Cat Pricing: Impressive Physics, Wrong Bottleneck"
date: 2026-04-02
categories: [techniques, catastrophe, pricing, actuarial]
tags: [quantum-computing, catastrophe-pricing, monte-carlo, qae, quantum-amplitude-estimation, nisq, cat-modelling, tail-risk, var, expected-shortfall, kirke, arXiv-2603-15664]
description: "Kirke (arXiv:2603.15664) applies Quantum Amplitude Estimation to catastrophe insurance tail-risk pricing and claims quadratic speedup over classical Monte Carlo. The maths is real. The practical relevance for UK cat pricing teams is close to zero, and the paper's own findings explain why."
math: true
author: burning-cost
---

Alexis Kirke's paper (arXiv:2603.15664, March 2026), presented at ICLR 2026's Financial AI workshop, applies Quantum Amplitude Estimation to catastrophe insurance Monte Carlo pricing. The claimed advantage: QAE converges at O(1/N) — quadratic speedup over the classical O(1/√N) — which would in principle allow far more accurate tail estimates with fewer model runs.

The convergence mathematics is real. Grover's algorithm does provide this speedup in theory. We are not disputing the quantum mechanics.

But the paper itself, in its own findings, identifies the problem with this research direction. And even setting that aside, the framing mistakes Monte Carlo convergence for the actual hard problem in catastrophe pricing.

---

## What the paper does

Kirke implements QAE using Qiskit Aer — IBM's quantum simulator running on classical hardware — and encodes lognormal catastrophe loss distributions as quantum oracles. The algorithm applies Grover amplification (up to 16 iterations) to estimate tail probabilities, validating against NOAA Storm Events data across 58,028 historical records.

Three findings. First: the oracle-model advantage is demonstrated in simulation — QAE does converge faster than classical Monte Carlo in the idealised case. Second: when the distribution can be solved analytically, classical closed-form solutions beat QAE on every metric. Third, and most importantly: **discretisation is the current bottleneck, not estimation.**

That third finding is buried in the results section. It deserves to be the abstract.

---

## The discretisation problem

To encode a probability distribution as a quantum oracle, you must discretise it. Continuous distributions become discrete approximations that can be loaded onto a quantum circuit. The precision of that discretisation is bounded by the number of qubits available. Current NISQ (Noisy Intermediate-Scale Quantum) devices have 50 to 1,000 noisy physical qubits. Fault-tolerant quantum computing — which is what you actually need for the theoretical speedup to materialise — requires thousands of error-corrected logical qubits, each composed of thousands of physical qubits.

The gap between where quantum hardware is today and where it needs to be for QAE to beat GPU-accelerated Monte Carlo is roughly three to four orders of magnitude in qubit count, and another order of magnitude in error rates. The paper validates entirely on a classical simulator, which sidesteps the noise problem entirely. On a real NISQ device, the Grover amplification degrades rapidly due to gate errors.

The paper's own Finding 3 says discretisation limits the method even in the best-case classical simulation. A perfect quantum computer would not fix this — you would still be approximating a continuous lognormal distribution with a discrete quantum state, and the approximation error would dominate the estimation advantage at reasonable precision levels.

This is not a criticism of the author's work. It is what the paper says. We just think it should be the lead finding rather than the third.

---

## The wrong bottleneck argument

Even if fault-tolerant quantum hardware arrived tomorrow, and even if the discretisation problem were solved, UK catastrophe pricing teams would not be the beneficiaries.

The actual hard problems in cat pricing are not Monte Carlo convergence problems.

Lloyd's managing agents and UK non-life property insurers doing cat pricing use vendor platforms — RMS Risk Intelligence, AIR Touchstone, Verisk Respond. These systems run millions of stochastic event iterations on GPU clusters and return aggregate EP curves in minutes. Computation is not the constraint. When a cat modeller asks for a faster simulation engine, they are not expressing a fundamental methodological problem. They are expressing impatience.

The genuine problems in UK cat pricing are:

**Hazard model accuracy.** How accurately does the windstorm or flood footprint model represent actual physical hazard at a given location? RMS and AIR have spent decades calibrating these models against observed events. For UK flood in particular, the Environment Agency's flood maps and the Lloyd's RDS scenarios are the binding constraints on accuracy, not the number of Monte Carlo iterations.

**Secondary uncertainty.** The spread of outcomes within a given event, driven by building stock variability, survey quality, and claims handling differences across insurers, is often larger than the primary hazard uncertainty. This is not a computational problem.

**Portfolio correlation.** Understanding which risks co-move in a given scenario — the correlation between marine, property, and BI losses in a major windstorm, for example — requires contract-level data integration and expert judgment. No amount of quantum speedup on individual risk distributions helps with this.

**Model risk.** The gap between vendor model outputs and actual loss experience varies significantly by peril, region, and carrier. Managing that gap requires historical back-testing and model blending, not faster convergence.

A quantum cat pricing tool that converges twice as fast to the wrong hazard model is not useful.

---

## The timeline

We are willing to entertain quantum computing as a genuine long-term technology watch item for actuarial modelling. The quadratic speedup is real in principle, and there are specific actuarial problems — high-dimensional integration, rare event probability estimation — where the advantage is theoretically meaningful.

But the realistic timeline for fault-tolerant quantum hardware capable of outperforming GPU clusters on Monte Carlo problems is 15 to 20 years, based on current roadmaps from IBM, Google, and IonQ. The UK cat market will face more pressing methodological challenges over that period: climate change trajectory uncertainty in flood and windstorm models, cyber accumulation modelling, and the integration of real-time IoT data into underwriting. None of these are waiting on quantum speedup.

---

## Our verdict

The paper is competently executed quantum computing research applied to an insurance problem. The QAE framework is correctly specified. The NOAA validation is appropriate. As a physics result, it is fine.

As guidance for UK cat pricing practice, it has nothing to offer in the near term. The paper's own findings tell you this — discretisation is the bottleneck, not estimation. And even if discretisation were solved, the actual bottlenecks in cat pricing are hazard model quality and correlation, not Monte Carlo convergence.

File this under technology watch. Revisit in 2040.
