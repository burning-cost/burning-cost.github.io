---
layout: post
title: "Spatial Error Correction vs Spatial Smoothing: Two Different Questions in Territory Rating"
date: 2026-03-26
categories: [spatial, territory-rating]
tags: [spatial, territory-rating, BYM2, spatial-autocorrelation, SEM, GBM, cochrane-orcutt, GMM, frequentist, bayesian, insurance-spatial, mboost, italian-insurance, postcode, motor, home]
description: "Balzer and Benlahlou's spatial GBM uses GMM pre-estimation and a Cochrane-Orcutt transformation to handle spatial autocorrelation in gradient boosting. It is a different tool from BYM2 — and not a substitute for it."
---

Two pricing teams, same problem: their territory model residuals fail a Moran's I test. Spatial autocorrelation is present. The GLM is producing correlated errors across adjacent postcode sectors, which means standard errors are wrong and the model is fitting noise rather than genuine spatial risk structure.

Both teams reach for a spatial method. One uses BYM2 — a Bayesian hierarchical model with intrinsic conditional autoregressive priors that smooths territory effects across neighbours. The other applies a spatial error model with gradient boosting — estimating spatial autocorrelation parameters via GMM, applying a Cochrane-Orcutt-type transformation to the data, and running a standard L2-boosting algorithm on the transformed dataset.

They are not solving the same problem. They might look like competing approaches to spatial autocorrelation, but the question each method is actually answering is different. Getting that distinction wrong will produce either an overcomplicated pricing model or a technically correct but commercially useless one.

---

## The question BYM2 answers

BYM2 is a spatial random effects model. The spatial structure is the point. You are trying to estimate territory relativities — area factors, postcode uplift, regional risk differentials — and you are using spatial smoothing to produce credible estimates in data-sparse areas.

The ICAR prior says: the territory effect for a postcode sector is drawn from a distribution centred on the average of its neighbours' effects. A thin sector with 20 claims borrows from the surrounding sectors that share its demographic and geographic characteristics. The result is a smoothed relativity surface with full posterior uncertainty per sector.

For a UK motor or home insurer running an annual territory review, this is almost certainly the right question. The product is the relativity surface. Spatial smoothing is the mechanism for getting a defensible number out of 50-claim sectors that would otherwise produce garbage estimates.

What BYM2 does not do well: it is a GLM. If your main model is a GBM with fifty covariates and you want spatial controls that live naturally inside the boosting framework, BYM2 is not a natural fit. And if you have multi-year panel data where the same postcode appears repeatedly, BYM2 is not set up to handle the panel structure.

---

## The question spatial GBM answers

Balzer and Benlahlou (arXiv:2603.14543, March 2026) extend model-based gradient boosting to spatial panel data with area effects. Their GSPECM framework — Generalized Spatial Panel model with Error Components — is a spatial error model. The spatial autocorrelation is a nuisance parameter, not the signal.

The algorithm runs in two stages:

**Stage 1 (GMM pre-estimation).** Before any boosting, estimate the spatial autocorrelation parameters rho1 and rho2 via Generalised Method of Moments, using the spatial weights matrix and initial residuals from a pilot model. This is computationally cheap and does not involve the boosting loop. The spatial weights matrix (k=10 nearest neighbours by geographic centroid, row-normalised) is researcher-specified, not learned.

**Stage 2 (Cochrane-Orcutt transformation + L2-boosting).** Apply a spatial filtering transformation to the outcome and covariates — analogous to the time-series Cochrane-Orcutt correction for serial autocorrelation, extended to the spatial dimension. The Mahalanobis distance loss function with the full spatial covariance matrix collapses, after transformation, to a standard L2 loss. Run `mboost` on the transformed data as normal.

The practical consequence: the spatial structure is absorbed in pre-processing. The boosting algorithm sees an L2 problem and runs identically to non-spatial mboost. Variable selection, early stopping via spatial cross-validation, and regularisation all work as usual. There is no custom loss function, no bespoke spatial boosting objective, just a smart data transformation that converts a spatial problem into a standard one.

This is elegant. It is also, critically, treating the spatial correlation as something to correct for rather than something to estimate. The output is a multi-covariate predictive model with spatial error correction, not a relativity surface.

---

## What the Italian insurance example shows

The paper's non-life insurance application uses Italian provinces as spatial units — 103 districts, multiple years of panel data, covariates including real GDP, bank deposits, judicial inefficiency, and regional indicators. After boosting with implicit variable selection, six variables are retained from twenty-one candidates. GDP elasticity on claim frequency: 0.47. Bank deposits (household income proxy): 0.13.

These are plausible macro-level results. GDP-rich areas have more insured assets and generate more claims. The model produces sensible economics.

What this does not demonstrate:

- Performance at postcode sector level. Italian provinces are large administrative units (average area ~2,900 km²). UK postcode sectors are 1-3 km² in urban areas. The spatial autocorrelation structure, the data sparsity problem, and the variance-bias tradeoff are all materially different at UK granularity.
- Superiority over BYM2 for territory rating. The paper does not benchmark against BYM2. It benchmarks against standard spatial panel regression and standard GBM, both of which it outperforms on prediction error. That is a lower bar.
- Applicability to the dominant UK spatial risk drivers. Flood and subsidence risk in UK home insurance have geographic patterns driven by geology, elevation, and river proximity — not macro-economic indicators. The Italian paper's covariate list is entirely economic. UK spatial risk structure is largely physical, which changes the variable selection problem entirely.

The Italian example is a proof of concept for the method, not a validation for UK pricing.

---

## The spatial cross-validation point

The most underappreciated contribution in the paper is not the Cochrane-Orcutt transformation. It is the spatial cross-validation used for early stopping.

Standard k-fold cross-validation assigns observations randomly to folds. For spatial data, this means the training set will contain areas adjacent to the validation set areas. If spatial autocorrelation is present — and you are building a spatial model precisely because it is — adjacent areas are correlated. The validation performance estimate is optimistic because the model can partially predict the validation fold from the spatially correlated training observations near it. You will stop too late and overfit.

Spatial cross-validation assigns areas to folds such that adjacent areas land in the same fold. The validation fold is then geographically separated from the training data. This is the correct approach for any spatial model: BYM2, spatial GBM, spatial GAM, or a standard GLM with postcode as a fixed effect.

Most UK pricing teams do not do this. Model selection for geographic predictors using random CV is a quietly widespread error.

---

## Why we are not building a library for this

The method lives in R as `github.com/micbalz/SpatPanelRegBoost`. It is not packaged. There is no Python implementation.

We considered whether this warranted a Python port into the Burning Cost toolset. It does not, for three reasons.

First, the use case does not arise often enough. The spatial error model is appropriate when you want spatial controls in a multi-covariate predictive model and your data has a panel structure. For UK P&C, most carriers have postcode-level panel data for home and motor. But "I have a GBM with thirty covariates and spatial panel data and my residuals fail Moran's I" is a fairly specific situation. The teams who encounter it are generally sophisticated enough to implement the GMM pre-step themselves using `libpysal` and a standard gradient boosting library.

Second, for the more common case — deriving territory factors — the right tool is BYM2, which we already have in [`insurance-gam`](/insurance-gam/). BYM2 produces a smoothed relativity surface with full posterior uncertainty per sector. That is what territory rating requires. Spatial GBM does not produce a relativity surface; it produces a multi-covariate predictive model with spatial error correction. These are different outputs.

Third, the "trick" is a preprocessing transformation, not a novel spatial learning algorithm. Implementing the Cochrane-Orcutt spatial transform and the GMM pre-step adds perhaps two hundred lines to an existing pipeline. That is not a library; it is a function. If you need it, write it.

---

## When to use which

**You want territory relativities — area factors for a rating table:** use BYM2 via `insurance-spatial`. You get a smoothed relativity surface, credibility-weighted estimates per sector, full posterior intervals, and year-on-year stability. The output is directly usable as a rating factor.

**You want to fit a GBM with spatial panel structure and multiple area-level covariates:** consider spatial GBM. The GMM pre-step handles autocorrelation, the Cochrane-Orcutt transformation reduces it to standard L2-boosting, and spatial cross-validation gives you honest early stopping. You will need to port the R implementation or build the preprocessing yourself.

**You want spatial controls in a single-snapshot model without panel structure:** a GAM with a spatial smooth (`mgcv::s(x, y, bs="gp")` in R, or `scikit-learn` with radial basis features) achieves the same goal with better tooling and wider documentation.

**Your spatial autocorrelation is entirely driven by area-level omitted variables that you will later include explicitly:** build a better feature set. Neither BYM2 nor spatial GBM is a substitute for including the actual drivers of geographic risk variation — flood zone, geology, crime rate, distance to coast. Spatial autocorrelation in residuals after including relevant features should be small.

The Balzer-Benlahlou paper is methodologically sound and fills a real gap. "How do I do spatial panel regression inside a GBM framework?" is a legitimate question with a clean answer. It is not, however, a new general-purpose territory pricing tool. The important conceptual contribution — treating spatial structure as something to correct for before modelling versus something to model directly — is worth understanding precisely because it clarifies when you should not reach for the spatial error approach.

---

**Paper:** Balzer, M. and Benlahlou, A. (2026). 'Gradient Boosting for Spatial Panel Models with Random and Fixed Effects.' arXiv:2603.14543. [https://arxiv.org/abs/2603.14543](https://arxiv.org/abs/2603.14543)

Related posts:

- [BYM2 Spatial Smoothing for Territory Ratemaking](/2026/02/23/spatial-territory-ratemaking-with-bym2/) — the ICAR prior, rho estimation, and why it scales to 9,000 postcode sectors
- [Getting Spatial Territory Factors Into Production](/2026/03/09/spatial-territory-ratemaking-bym2/) — the full pipeline from adjacency matrix to Emblem export
- [BYM2 Territory Modelling: Posterior Uncertainty and Year-on-Year Stability](/2026/03/15/your-territory-model-ignores-spatial-autocorrelation/) — why the posterior interval is the actual product
- [Spatial Panel GBMs: A Better Way to Price Geography](/2026/03/26/spatial-panel-gbm-geographic-insurance-pricing/) — detailed breakdown of arXiv:2603.14543 with Python code sketch
