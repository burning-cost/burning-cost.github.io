---
layout: page
title: "Insurance Fairness and Proxy Discrimination Testing"
description: "Proxy discrimination auditing, discrimination-free pricing via optimal transport, and FCA Consumer Duty documentation for UK insurance pricing models. Python library."
permalink: /topics/fairness-proxy-discrimination/
---

The FCA's Consumer Duty and EP25/2 have sharpened regulatory attention on pricing fairness. The obligation is not simply to avoid using protected characteristics directly — that was already the case under the Equality Act 2010. The harder problem is proxy discrimination: a rating factor that is genuinely predictive of risk but also highly correlated with a protected characteristic can produce systematically different premiums for protected groups without the protected variable ever appearing in the model.

Detecting proxy discrimination requires testing the model's pricing distribution against an external reference — or, more rigorously, testing whether the pricing relativities for a suspect variable can be explained by risk differences alone, or whether they reflect the variable's correlation with a protected characteristic. Correcting it requires either removing the proxy or applying an optimal transport adjustment that preserves the risk signal while eliminating the discriminatory component. Both of these steps need to be documented in a way that would survive FCA scrutiny.

The [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) library implements the full audit workflow: correlation decomposition to identify suspect proxies, LRTW-based discrimination testing, optimal transport discrimination-free pricing, and structured output for Consumer Duty documentation. It works with and without sensitive attribute data — for cases where you do not hold protected characteristics directly.

**Library:** [insurance-fairness on GitHub](https://github.com/burning-cost/insurance-fairness) &middot; `pip install insurance-fairness`

---

## Tutorials and introductions

- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — the foundational tutorial: detecting postcode as a proxy and applying discrimination-free pricing
- [FCA Consumer Duty Pricing Fairness in Python](/2026/03/20/fca-consumer-duty-pricing-fairness-python/) — end-to-end Consumer Duty compliance workflow using `insurance-fairness`
- [FCA Proxy Discrimination Testing in Python: A Practical Guide](/2026/03/22/fca-proxy-discrimination-python-testing-guide/) — step-by-step guide to EP25/2-aligned proxy testing with documentation output

---

## Techniques and methods

- [Discrimination-Free Pricing in Python: Causal Paths, Optimal Transport, and the FCA](/2026/03/10/insurance-fairness-ot/) — optimal transport adjustment for discrimination-free premiums while preserving risk signal (now in [insurance-fairness](https://github.com/burning-cost/insurance-fairness) as `insurance_fairness.optimal_transport`)
- [Three Ways to Detect Proxy Discrimination in Your Pricing Model: When They Agree and When They Don't](/2026/03/22/three-methods-proxy-discrimination-detection-compared/) — LRTW, correlation decomposition, and causal path testing compared on the same data
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) — proxy testing without holding gender or ethnicity data
- [Discrimination-Insensitive Pricing: Beyond Removing the Protected Variable](/2026/03/25/discrimination-insensitive-pricing/) — the limitations of simply dropping a correlated feature and what to do instead
- [Fair Pricing in Long-Term Insurance: What arXiv:2602.04791 Means for UK Actuaries](/2026/03/25/fair-pricing-long-term-insurance/) — multi-state fairness for income protection and PMI under the pure protection market study
- [Multi-Objective Pricing: The Pareto Front of Fairness and Profit](/2026/03/25/multi-objective-pricing-pareto-front-fairness-profit/) — optimising rate change across fairness constraints and loss ratio targets simultaneously

---

## Benchmarks and validation

- [Does proxy discrimination testing actually work?](/2026/03/28/does-proxy-discrimination-testing-actually-work/) — sensitivity and specificity of detection methods on synthetic data with known discrimination
- [Does Proxy Discrimination Testing Actually Work?](/2026/03/28/does-proxy-discrimination-testing-actually-work/) — extended empirical validation across different proxy strengths and portfolio sizes

---

## Library comparisons

- [EquiPy vs insurance-fairness: Two Different Questions About Fairness in Insurance Pricing](/2026/03/19/equipy-vs-insurance-fairness/) — EquiPy's post-processing approach versus causal discrimination testing
- [Fairlearn vs insurance-fairness: Why Generic ML Fairness Tools Miss What the FCA Cares About](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) — Fairlearn optimises for statistical parity; the FCA cares about proxy causality

---

## Regulatory context

- [The FCA Is Investigating Home and Travel Insurers](/2026/03/19/the-fca-is-investigating-home-and-travel-insurers/) — what the investigation means for pricing team responsibilities
