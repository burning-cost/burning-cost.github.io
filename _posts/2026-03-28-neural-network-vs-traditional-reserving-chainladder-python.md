---
layout: post
title: "Reserving with chainladder-python Part 3: Neural Networks vs Traditional Methods"
date: 2026-03-28
categories: [tutorials]
tags: [reserving, IBNR, chain-ladder, neural-networks, DeepTriangle, Bornhuetter-Ferguson, bootstrap-ODP, chainladder, python, stochastic-reserving, tutorial]
description: "When does it make sense to reach beyond chain ladder and bootstrap ODP for neural reserving methods? We compare DeepTriangle, individual RNN approaches, and the Richman-Wüthrich one-shot framework against traditional aggregate triangle methods — and give you an honest assessment of where each belongs."
seo_title: "Neural Network vs Traditional Reserving in Python: DeepTriangle vs Chain Ladder and Bootstrap ODP"
---

[Part 1](/2026/04/13/chain-ladder-python-reserving-tutorial/) of this series covered the chain ladder method in `chainladder`: how to fit development factors, extract IBNR, and read a loss triangle in Python. [Part 2](/2026/04/15/stochastic-reserving-python-bootstrap-odp/) added the stochastic layer — Mack analytical errors and Bootstrap ODP simulation — so you could produce a full IBNR distribution for Solvency II SCR and IFRS 17 risk adjustment.

For the vast majority of reserving exercises, you can stop there. Chain ladder plus Bootstrap ODP is well-understood, fast to run, auditable, and defensible to a regulator. The question this post addresses is narrow but important: **under what circumstances does a neural network approach actually add something?**

The honest answer is: less often than the academic literature implies, but there are real cases where it matters. We will work through what those cases look like.

---

## The traditional methods: what you already have

Before introducing the alternatives, it is worth being precise about what "traditional" means. There are three methods in serious use in the UK market.

**Chain ladder** applies volume-weighted development factors across accident years to project the latest diagonal to ultimate. The method assumes that the pattern of development is consistent across accident years — a restriction that is often violated by changing settlement practice, inflation, or portfolio mix shifts. `chainladder` implements this as `Chainladder().fit(development_triangle)`. We covered this in Part 1.

**Bornhuetter-Ferguson** blends the chain ladder projection with an *a priori* expected loss ratio, weighting each by the credibility of the development data. For immature accident years — where you have seen little of the ultimate development — BF places more weight on the prior and less on the (sparse) triangle data. In `chainladder`, `BornhuetterFerguson` takes `sample_weight` (premium or exposure per origin year) and `apriori` (the ELR scalar):

```python
import chainladder as cl

# Apply the pandas 3.x valuation fix first — see Part 2 for full details
_orig_valuation = cl.Triangle.valuation.fget
cl.Triangle.valuation = property(lambda self: _orig_valuation(self).floor("us"))

raa = cl.load_sample('raa')
dev = cl.Development().fit_transform(raa)

# sample_weight is premium per origin; apriori is the ELR
# Here we use the CL ultimate as a reasonable premium proxy
cl_ultimate = cl.Chainladder().fit(dev).ultimate_

bf = cl.BornhuetterFerguson(apriori=0.60).fit(dev, sample_weight=cl_ultimate)
print(f"BF IBNR: {bf.ibnr_.sum():,.0f}")
```

BF is the right approach when the most recent one or two accident years are immature. It is one of the most underused methods in `chainladder` and should be in every practitioner's default workflow alongside the chain ladder.

**Bootstrap ODP** (covered in Part 2) resamples Pearson residuals from the chain ladder fit to generate an empirical IBNR distribution. The 99.5th percentile from 5,000 simulations gives you your reserve risk SCR. This is standard practice at most UK non-life insurers under Solvency II.

These three methods share one fundamental assumption: they operate on *aggregate triangles*. They compress every claim in an accident year into a single number per development period. Everything that happens at claim level — the handler's reserve estimate, payment sequence, case status, reporting delay — is discarded.

---

## The neural network alternatives

### DeepTriangle (Kuo, 2019)

Kevin Kuo's DeepTriangle paper ([arXiv:1804.09253](https://arxiv.org/abs/1804.09253)) is the clearest published attempt to apply sequence modelling directly to loss triangles. The central insight is that the rows of a loss triangle are sequences of partial development observations — exactly the sort of data that recurrent neural networks were designed for.

The architecture is straightforward to describe even without running TensorFlow:

```
Input: for each accident year, a sequence of paid loss ratios
       observed up to the latest diagonal, zero-padded
       to a fixed maximum development length.

Encoder: a GRU (Gated Recurrent Unit) that reads the sequence
         left-to-right and produces a single hidden state
         summarising the development pattern so far.

Decoder: a GRU that unrolls forward, predicting one future
         development period at a time, conditioned on the
         encoded hidden state.

Output: predicted paid loss ratios for all future development
        periods, from which IBNR is reconstructed.
```

Kuo trained this on the CAS Loss Reserve Database — 50 US property-casualty companies across multiple lines, giving around 200 triangles. The key point is the **training data requirement**: DeepTriangle learns development patterns across companies, not from a single triangle. A single RAA triangle (10 accident years × 10 development periods = 100 cells) gives the model almost nothing to train on.

Results from the paper: on held-out test companies, DeepTriangle outperformed chain ladder on around 60% of company-line combinations by mean absolute error. But the margin was modest, and chain ladder remained competitive on short-tail lines and well-behaved development patterns.

### Individual claims RNN approaches

A different strand of research applies sequence models at the *claim* level rather than the *triangle* level. Each claim generates a sequence of payment observations, and an RNN learns to predict the ultimate payment from whatever development history is available.

Wüthrich (2018) laid out the theoretical framework ("Machine Learning in Individual Claims Reserving", *Scandinavian Actuarial Journal*). The LSTM architecture reads a claim's payment history as a time series and outputs a predicted ultimate. Gabrielli et al. (2020) ("An Individual Claims History Simulation Machine") developed a simulation framework for generating realistic synthetic claim histories at scale, which has become the standard benchmark dataset for evaluating these models.

The practical architecture is approximately:

```python
# Conceptual only — illustrative, not runnable
# For individual claim i with payment history [p_1, ..., p_t]:

# Input: sequence of (payment, status, features) tuples
# LSTM encoder: 64-128 hidden units, 2-3 layers
# Output head: single neuron predicting log(ultimate - paid_to_date)

# Key difference from DeepTriangle: model operates at claim level,
# trained on a book with 50,000+ historical closed claims.
```

Richman and Wüthrich (arXiv:2603.11660) took this further with a "one-shot" approach that bypasses iterative forward-stepping: a single projection from current paid to ultimate, analogous to a claim-level development factor. We covered that paper in detail [here](/2026/03/25/one-shot-individual-claims-reserving-neural-networks-vs-chain-ladder/). Their key finding — that linear regression on claim features matches the neural network on small datasets — is important context for what follows.

---

## The honest comparison

### Where traditional methods remain the right answer

For a standard UK motor or household portfolio, you have a triangle with perhaps 10-15 accident years, quarterly or annual development periods, and aggregate paid or incurred data. Under these conditions:

**Chain ladder + Bootstrap ODP is sufficient.** It is transparent, fast, requires no training data beyond the triangle itself, and can be fully explained to a reserving committee or external auditor. The bootstrap gives you a credible reserve range without any assumptions about the shape of the development distribution.

Neural approaches do not improve on this combination unless several conditions hold simultaneously.

### When neural approaches actually add something

**Condition 1: You have a large training set.** DeepTriangle was trained on ~200 triangles from 50 companies. An individual claims RNN requires tens of thousands of fully-developed historical claims. If you are reserving a single UK employer's liability book with 8 accident years, you do not have this. If you are a large Lloyd's syndicate with 15 years of granular claims data across 40,000+ claims per accident year, you might.

**Condition 2: Development patterns are heterogeneous across cohorts.** Chain ladder assumes that the age-to-age factors are stable across accident years. If your settlement practice changed in 2020, your inflation profile shifted after 2022, or your mix of claim types has moved materially, the historical development pattern is a biased guide to future development. A neural model trained on claim features can, in principle, condition on these changes. In practice, you need enough post-change data to train on.

**Condition 3: Claim-level features carry predictive signal.** The individual RNN approaches only beat chain ladder when the claim-level data — handler reserves, reporting delay, claim type, payment history — genuinely predicts variation in ultimate development. For short-tail lines where claims close within 12 months, there is limited information in the payment sequence that aggregate triangles do not already capture. Long-tail liability, bodily injury, and professional indemnity claims have rich development trajectories that make individual modelling worthwhile.

**Condition 4: You can validate the model out-of-sample.** A neural reserving model that cannot be tested against held-out accident years is not deployable in a regulated environment. For a book with 10 accident years, there are not enough rows to hold out a meaningful test set while retaining a representative training set. The minimum viable validation setup requires at least 20 accident periods.

---

## A calibration exercise on the RAA triangle

To make this concrete, here is what the traditional methods produce on the RAA dataset. All three methods run in under a second. The pandas timestamp fix from Part 2 is required for `chainladder` 0.9.1 with pandas 3.x.

```python
import chainladder as cl
import numpy as np

# Valuation timestamp fix — see Part 2 for explanation
_orig_valuation = cl.Triangle.valuation.fget
cl.Triangle.valuation = property(lambda self: _orig_valuation(self).floor("us"))

raa = cl.load_sample('raa')
dev = cl.Development().fit_transform(raa)

# Chain ladder
cl_model = cl.Chainladder().fit(dev)
cl_ibnr = cl_model.ibnr_.sum()

# Bornhuetter-Ferguson — use CL ultimate as the a priori premium base
bf_model = cl.BornhuetterFerguson(apriori=0.60).fit(dev, sample_weight=cl_model.ultimate_)
bf_ibnr = bf_model.ibnr_.sum()

# Bootstrap ODP (5,000 simulations)
boot = cl.BootstrapODPSample(n_sims=5000, random_state=42)
boot.fit(dev)
sims = boot.transform(dev)
total_ibnr = cl.Chainladder().fit(sims).ibnr_.sum(axis=2).values.flatten()
total_ibnr = total_ibnr[~np.isnan(total_ibnr)]

print(f"Chain ladder IBNR:         {cl_ibnr:>10,.0f}")
print(f"BF IBNR (60% ELR):         {bf_ibnr:>10,.0f}")
print(f"Bootstrap ODP 50th pctile: {np.percentile(total_ibnr, 50):>10,.0f}")
print(f"Bootstrap ODP 75th pctile: {np.percentile(total_ibnr, 75):>10,.0f}")
print(f"Bootstrap ODP 99.5th pct:  {np.percentile(total_ibnr, 99.5):>10,.0f}")
```

The RAA triangle has 10 rows. DeepTriangle would require you to pool it with hundreds of similar triangles to produce a usable training set. An individual claims RNN is simply inapplicable — the RAA data is aggregate paid losses, not a claims register.

This is not a pathological edge case. It is the normal situation for a mid-sized UK insurer producing annual reserve estimates.

---

## What the academic results actually show

It is worth being direct about the evidence base.

Kuo (2019) showed that DeepTriangle outperforms chain ladder on a pooled dataset of US commercial lines triangles. The improvement is real but moderate — the mean absolute percentage error improvement is around 10-15% on the test set. That result applies to a setting where you have 200 triangles to train and test on. It does not generalise to a single-company reserving exercise.

Wüthrich (2018) and Gabrielli et al. (2020) demonstrate that individual claims models can outperform aggregate triangle methods when the claim-level feature set is rich and the training data is large. Their simulation machine generates synthetic claims with realistic payment trajectories, which is valuable for building benchmarks. But the improvement is conditional on data quality and volume that most UK portfolios do not have.

The Richman-Wüthrich one-shot paper (2026) finds that, on datasets of 20,000-60,000 claims, a simple linear regression on claim features already closes most of the gap between chain ladder and neural models. The neural network adds a further increment, but it is not the dominant effect. The dominant effect is using claim-level features at all — not the particular machine learning architecture applied to them.

---

## The decision framework

The practical question is not "traditional or neural?" but "what data do I have and what problem am I solving?"

| Situation | Recommended method |
|---|---|
| Standard triangle, 8-15 accident years, aggregate data | Chain ladder + Bootstrap ODP |
| Immature recent years (≤3 development periods observed) | Bornhuetter-Ferguson for latest years, chain ladder elsewhere |
| Development pattern instability suspected | BF with expert-adjusted a priori, sensitivity test via bootstrap |
| Large book, 50,000+ claims per year, long-tail line | Individual claims model worth piloting — validate out-of-sample |
| Multiple portfolios or companies, homogeneous lines | DeepTriangle-style pooled training viable |
| Regulatory capital quantification (SII SCR / IFRS 17 RA) | Bootstrap ODP is the industry standard; neural models are not yet accepted by most internal model validators |

The last row matters. Even if a neural approach produces better point estimates, the bootstrap ODP has a 25-year track record of regulatory acceptance and a transparent residual-resampling methodology that can be explained step by step. Replacing it with a GRU decoder for capital purposes requires a model validation exercise of a different order of magnitude.

---

## What `chainladder` gives you

The `chainladder` library covers the methods that matter for the vast majority of reserving exercises:

- `Chainladder` — volume-weighted development factors
- `BornhuetterFerguson` — blended projection with a priori
- `Benktander` — iterated BF, converges toward chain ladder
- `CapeCod` — parameterises the ELR from the triangle itself
- `MackChainladder` — analytical standard errors
- `BootstrapODPSample` — full simulation reserve distribution

There is no neural network method in `chainladder`. That is the right call for a production reserving library. The traditional methods have formal actuarial standards behind them (IFoA Technical Actuarial Standards, CAS ASTIN papers), have been validated against real-world reserve movements for decades, and integrate cleanly with Solvency II internal model frameworks.

If you want to pilot a neural approach alongside your existing triangle work, the entry point is the [CAS Loss Reserve Database](https://www.casact.org/research/reserve_data/) for training data, Kuo's [DeepTriangle repository](https://github.com/kevinykuo/deeptriangle) for a reference implementation, and the Gabrielli et al. simulation machine for generating synthetic individual claims training data.

---

## Conclusion

Chain ladder and Bootstrap ODP remain the right tools for most UK reserving exercises — not because they are old, but because the data requirements of the neural alternatives are genuinely demanding. You need either a large panel of comparable triangles (for DeepTriangle-style approaches) or a large book of individually-tracked claims with rich feature data (for individual claims neural models). Most portfolios do not have either.

Where the case for neural methods is strongest — long-tail liability, large books, granular claims data — the dominant gain comes from using claim-level features rather than from the neural architecture itself. A well-specified linear regression on claim status, incurred, and reporting delay will close most of the gap. The neural model is a refinement on top of that.

The practical agenda for most reserving actuaries is:
1. Get the chain ladder right — wash triangles, check development factor stability, exclude outlier accident years
2. Add BF for immature years
3. Run Bootstrap ODP for the reserve distribution and capital quantities
4. If you have a large long-tail book, pilot an individual claims model — but treat it as supplementary validation, not a replacement, until you have tested it out-of-sample across at least three held-out accident years

Parts [1](/2026/04/13/chain-ladder-python-reserving-tutorial/) and [2](/2026/04/15/stochastic-reserving-python-bootstrap-odp/) of this series give you the code for steps 1-3. That is where the majority of reserving improvement is available.

---

*Further reading: Kuo (2019) "DeepTriangle: A Deep Learning Approach to Loss Reserving" ([arXiv:1804.09253](https://arxiv.org/abs/1804.09253)); Wüthrich (2018) "Machine Learning in Individual Claims Reserving", Scandinavian Actuarial Journal; Gabrielli, Richman and Wüthrich (2020) "An Individual Claims History Simulation Machine", Risks 8(2); Richman and Wüthrich (2026) "From Chain-Ladder to Individual Claims Reserving" ([arXiv:2602.15385](https://arxiv.org/abs/2602.15385)).*
