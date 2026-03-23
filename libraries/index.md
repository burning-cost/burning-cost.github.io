---
layout: page
title: "Library Documentation"
description: "Detailed documentation for the 10 flagship Burning Cost libraries — installation, quick start, API reference, and links to blog posts."
permalink: /libraries/
---

Ten libraries for the problems that matter most in UK non-life insurance pricing. Each has a documentation page with installation instructions, working code examples, and an API reference drawn from the actual source.

| Library | Problem it solves | Category |
|---------|-------------------|----------|
| [insurance-fairness](/insurance-fairness/) | Proxy discrimination auditing — FCA Consumer Duty evidence | Fairness & Regulation |
| [insurance-governance](/insurance-governance/) | PRA SS1/23 model validation and MRM governance packs | Validation & Governance |
| [insurance-causal](/insurance-causal/) | Double machine learning — remove confounding bias from rating factors | Causal Inference |
| [insurance-conformal](/insurance-conformal/) | Distribution-free prediction intervals with finite-sample coverage guarantees | Uncertainty Quantification |
| [insurance-monitoring](/insurance-monitoring/) | Model drift detection, calibration monitoring, and champion/challenger A/B testing | Monitoring |
| [insurance-whittaker](/insurance-whittaker/) | Whittaker-Henderson smoothing for experience rating tables | Smoothing |
| [insurance-telematics](/insurance-telematics/) | Raw telematics trip data to GLM-ready risk scores via HMM | Telematics |
| [insurance-credibility](/insurance-credibility/) | Bühlmann-Straub group credibility and Bayesian experience rating | Credibility |
| [insurance-frequency-severity](/insurance-frequency-severity/) | Sarmanov copula joint frequency-severity modelling | Model Building |
| [insurance-gam](/insurance-gam/) | Interpretable GAMs — EBM, Neural Additive Models, Pairwise Interaction Networks | Model Building |

All libraries are MIT-licensed, on PyPI, and built for Python 3.10+. The GitHub organisation is [burning-cost](https://github.com/burning-cost).

---

## Why these ten

The rest of the portfolio — 24 more libraries — covers important problems. These ten are the ones where no adequate Python solution existed before, where the methodology is genuinely differentiated, or where the compliance stakes are high enough that getting it right matters beyond just accuracy.

A UK pricing team doing its job properly needs all of them at some point. A team starting from scratch should begin with governance (know what a valid model looks like), fairness (Consumer Duty is not optional), and conformal prediction (give your point estimates uncertainty bounds before putting them in front of a pricing committee).

---

## Full library catalogue

The [Tools page](/tools/) lists all 34 libraries with descriptions and install commands. Use it if you are looking for something outside the flagship ten — spatial ratemaking, GLM interaction detection, distributional GBMs, survival models, and more.
