# burning-cost.github.io

![License: MIT](https://img.shields.io/badge/license-MIT-green)

Insurance pricing education and open-source tooling for UK actuaries and pricing teams.

The site at [burning-cost.github.io](https://burning-cost.github.io) publishes worked examples, methodology explainers, and links to all open-source libraries.

## Flagship Libraries

Ten libraries we consider genuinely differentiated — tools for hard problems in UK pricing where no adequate open-source Python solution existed before.

**Regulatory compliance**

| Library | Problem it solves |
|---------|------------------|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing aligned to FCA Consumer Duty and Equality Act 2010 |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | PRA SS1/23 model validation reports: bootstrap Gini CI, A/E CI, double-lift, renewal cohort test |

**Causal inference**

| Library | Problem it solves |
|---------|------------------|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double machine learning to deconfound rating factors where channel and behaviour bias standard GLM coefficients |

**Uncertainty quantification**

| Library | Problem it solves |
|---------|------------------|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals with finite-sample coverage guarantees; Solvency II SCR bounds |
| [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) | GAMLSS for Python: model mean, dispersion, shape, and zero-inflation as functions of covariates |

**Smoothing and experience rating**

| Library | Problem it solves |
|---------|------------------|
| [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | Whittaker-Henderson 1D/2D smoothing with REML lambda selection and Bayesian credible intervals |
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Bühlmann-Straub credibility in Python with mixed-model equivalence — caps thin segments, stabilises NCD factors |

**Telematics**

| Library | Problem it solves |
|---------|------------------|
| [insurance-telematics](https://github.com/burning-cost/insurance-telematics) | HMM driving state classification from raw 1Hz GPS/accelerometer data to GLM-compatible risk scores |

**Advanced modelling**

| Library | Problem it solves |
|---------|------------------|
| [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) | Sarmanov copula joint frequency-severity with analytical premium correction — tests the independence assumption every model makes |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | EBM and Neural Additive Model: transparency of a GLM, predictive power of a neural network |

## Full Library Portfolio

36 libraries in total, covering the full pricing workflow from data validation through to deployment and monitoring. See [burning-cost.github.io/tools](https://burning-cost.github.io/tools/) for the complete list.

## Site structure

Built with Jekyll. Posts are in `_posts/`. The `course/` directory contains structured learning materials.
