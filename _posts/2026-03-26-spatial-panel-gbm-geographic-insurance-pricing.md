---
layout: post
title: "Spatial Panel GBMs: A Better Way to Price Geography"
date: 2026-03-26
categories: [pricing, techniques, research]
tags: [spatial-pricing, gbm, panel-data, geographic-pricing, area-factors, postcode, BYM2, CAR, spatial-autocorrelation, mboost, italian-insurance, home-insurance, motor-insurance, blier-wong, credibility, uk-motor, python, arXiv]
description: "Balzer and Benlahlou (arXiv:2603.14543) extend gradient boosting to spatial panel data. Here is what it does, how it compares to BYM2 and Blier-Wong, and when a UK pricing team would actually reach for it."
---

Geographic pricing is one of those problems that looks solved from the outside — you have postcodes, you have claims, you build an area factor. In practice, it is almost never solved cleanly. The claims data is thin at small areas, spatial autocorrelation means your postcode factors are correlated in ways that GLM independence assumptions do not capture, and if your book spans multiple years the same postcode appears repeatedly in the panel, which is not the same as independent observations.

Balzer and Benlahlou (March 2026, arXiv:2603.14543) propose extending model-based gradient boosting to handle exactly this structure: spatial panel data with random or fixed area effects. The paper is worth understanding even if you are not going to implement it this month, because it clarifies what the current alternatives actually assume — and where those assumptions quietly fail.

---

## Why geographic pricing is genuinely hard

The three problems are distinct but tend to arrive together.

**Spatial autocorrelation.** Claims in adjacent postcodes are not independent. Flood risk, crime rates, subsidence geology, and commuting patterns all have geographic clustering. If you fit a GLM with postcode as a factor, the residuals will be spatially correlated — the model is not capturing the true structure, and standard errors are optimistic. Area factors estimated this way undersmooth in low-exposure areas and oversmooth globally.

**Thin areas.** A UK home insurer might have 1,200 postcode sectors in the rating territory. Many rural ones will have fewer than 50 claims over a five-year period. Maximum likelihood on 50 claims gives you area factors with confidence intervals wide enough to be almost meaningless. The credibility-weighted answer — shrink toward the regional mean — is defensible but ad hoc. You are choosing the shrinkage parameter by feel rather than by estimating it from the data.

**Panel structure.** If you pool five years of data at postcode sector level, the same area appears five times. That is not five independent observations; it is one area with five years of correlated experience. Treating it as five rows in a GLM understates the within-area persistence and overstates your effective sample size.

The standard UK approaches are: postcode-level GLM with area as a fixed effect (simple, ignores spatial correlation and panel structure), spatial CAR/BYM2 priors in a Bayesian GLM (handles autocorrelation well, difficult to extend to GBM, slow at scale), or Blier-Wong geographic embeddings (handles zero-exposure areas, requires embedding infrastructure) — for a deep-dive on why standard postcode factor methods understate spatial uncertainty, see [your territory model ignores spatial autocorrelation](/2026/03/15/your-territory-model-ignores-spatial-autocorrelation/). Each solves one or two of the three problems.

---

## What Balzer and Benlahlou propose

The paper extends the `mboost` framework — model-based gradient boosting — to spatial panel models. The setup is a linear predictor with three components: covariates (the standard rating factors), individual area effects (random or fixed), and a spatial error term that captures the remaining geographic correlation after conditioning on the area effects.

The boosting algorithm iterates over base-learners, selecting at each step the learner that best reduces the current negative gradient. What changes relative to standard GBM is that the base-learners include components for the panel structure. In the random effects case, the full covariance matrix — which incorporates spatial autocorrelation — enters the objective. In the fixed effects case, a Cochrane-Orcutt-type spatial transformation is applied to the data before boosting, eliminating the time-invariant area effects and converting the problem to a standard L2 loss.

The spatial weights matrix is required at setup time (it encodes which areas are adjacent), but the spatial structure is then absorbed by the transformation rather than re-estimated at every boosting step. This means the algorithm is not avoiding the spatial weights matrix — it still needs it — but it is not pre-specifying the *magnitude* of spatial effects. That is learned through the boosting iterations.

Variable selection is a byproduct: base-learners that do not reduce the gradient get selected less often, so irrelevant features are deselected automatically. The paper uses 5-fold spatial cross-validation to choose the stopping iteration, which is the right approach for spatial data — standard random CV would leak spatial correlation between training and validation folds.

The Italian non-life insurance application is the most relevant empirical result for insurance readers. The paper fits a spatial panel model across Italian districts with real GDP, bank deposits, and judicial inefficiency as covariates for claim frequency. After boosting with deselection, six variables are retained from twenty-one candidates. GDP elasticity is estimated at 0.47, household income (proxied by bank deposits) at 0.13. These are plausible numbers: areas with higher economic activity have more insured assets and higher claim frequencies.

---

## How this compares to what UK teams do

**BYM2 (spatial CAR in a Bayesian GLM).** The Besag-York-Mollié model, updated to the Riebler et al. (2016) reparameterisation, is probably the most principled approach for home pricing at postcode sector. It handles spatial autocorrelation through intrinsic conditional autoregressive priors and deals sensibly with zero-exposure areas. Its weaknesses: it is a GLM, so it inherits all GLM limitations for non-linear relationships; it does not naturally handle the panel structure; and fitting it with Stan or INLA on a large portfolio is slow. It also requires the adjacency structure as input — the same spatial weights matrix issue the Balzer-Benlahlou paper faces.

The spatial panel GBM has one material advantage over BYM2: it handles high-dimensional covariates and non-linear relationships natively. If your geographic pricing model has thirty covariates plus area effects, BYM2 becomes unwieldy and requires strong priors on every coefficient. GBM handles this through regularisation-by-selection.

**Blier-Wong geographic embeddings.** The 2022 ASTIN paper trains spatial embeddings — analogous to word2vec — for geographic units using demographic and infrastructure characteristics, then feeds these embeddings as features into a GLM. The key advantage over BYM2 is that it generates embeddings for zero-exposure territories using their geographic characteristics alone — no claims needed. The disadvantage is the embedding training infrastructure and the harder regulatory explanation. The spatial panel GBM and Blier-Wong are solving adjacent problems: Blier-Wong is primarily about representing spatial structure in features; Balzer-Benlahlou is primarily about the model structure for panel data. They are not direct substitutes.

**GLM postcode offsets.** The simplest approach: fit a GLM with postcode sector as a fixed effect using claims data, smooth the resulting factors, and apply as an offset in the main model. This is the baseline that most UK carriers use. It completely ignores spatial autocorrelation in estimation, treats multiple years from the same postcode as independent, and requires smoothing as a separate manual step. It works adequately for large books with dense postcode coverage. At postcode district level for rural areas it falls apart.

---

## A Python sketch of the structure

The R `mboost` package implements the Balzer-Benlahlou framework. Python lacks a direct equivalent, but the structure of a spatial panel GBM with area effects can be sketched using `lightgbm` with a manual panel transformation and a spatial regularisation offset:

```python
import numpy as np
import pandas as pd
import lightgbm as lgb

# Assume: df with columns area_id, year, covariates..., claims, exposure
# W: sparse adjacency matrix for area_id (n_areas x n_areas)

def cochrane_orcutt_spatial(df, W, rho=0.4):
    """
    Apply spatial Cochrane-Orcutt transformation to eliminate
    area fixed effects from the panel. rho is the spatial AR parameter.
    Pre-estimate rho from a pilot GLM residual; here we fix it for illustration.
    """
    areas = df["area_id"].unique()
    n = len(areas)
    I = np.eye(n)
    S = I - rho * W  # spatial filtering matrix

    transformed = []
    for year, grp in df.groupby("year"):
        grp = grp.set_index("area_id").reindex(areas)
        # Apply spatial filter to the outcome and covariates
        y = grp["log_freq"].values
        X = grp[covariates].values
        y_star = S @ y
        X_star = S @ X
        transformed.append(
            pd.DataFrame(X_star, columns=covariates)
            .assign(y_star=y_star, area_id=areas, year=year)
        )
    return pd.concat(transformed, ignore_index=True)


# Spatial cross-validation: hold out spatially contiguous blocks
def spatial_cv_split(df, W, n_folds=5, seed=42):
    """
    Assign areas to folds such that adjacent areas tend to land
    in the same fold — prevents spatial leakage between train/val.
    Uses a greedy graph colouring heuristic.
    """
    rng = np.random.default_rng(seed)
    areas = list(df["area_id"].unique())
    area_to_idx = {a: i for i, a in enumerate(areas)}
    area_fold = {}
    for a in rng.permutation(areas):
        i = area_to_idx[a]
        neighbour_folds = {
            area_fold[areas[j]] for j in np.where(W[i, :] > 0)[0]
            if areas[j] in area_fold
        }
        candidates = set(range(n_folds)) - neighbour_folds
        area_fold[a] = int(rng.choice(sorted(candidates))) if candidates else int(rng.integers(n_folds))
    df["fold"] = df["area_id"].map(area_fold)
    return df


# Fit GBM on transformed data with early stopping over spatial folds
covariates = ["log_gdp", "bank_deposits", "judicial_index", "urban_density"]

df_star = cochrane_orcutt_spatial(df, W, rho=0.4)
df_star = spatial_cv_split(df_star, W)

val_fold = 0
train_data = lgb.Dataset(
    df_star.loc[df_star["fold"] != val_fold, covariates],
    label=df_star.loc[df_star["fold"] != val_fold, "y_star"],
)
val_data = lgb.Dataset(
    df_star.loc[df_star["fold"] == val_fold, covariates],
    label=df_star.loc[df_star["fold"] == val_fold, "y_star"],
    reference=train_data,
)

params = {
    "objective": "regression",
    "learning_rate": 0.1,   # matches mboost default step size s=0.1
    "num_leaves": 4,        # shallow trees: each iteration = weak base-learner
    "min_data_in_leaf": 20,
    "verbose": -1,
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)],
)

print(f"Best iteration: {model.best_iteration}")
print(f"Validation RMSE: {model.best_score['valid_0']['l2']**0.5:.4f}")
```

This is not a faithful implementation of the Balzer-Benlahlou algorithm — the paper's random effects version uses a full Mahalanobis-norm objective that standard GBMs do not expose. But it captures the key ideas: pre-transform the panel data to absorb the spatial structure, use shallow base-learners with a small step size, and use spatially-blocked cross-validation to choose the stopping point.

The R `mboost` package is the right tool if you want the full algorithm. The paper's supplementary material provides the full estimation code.

---

## When you would actually use this

**Use spatial panel GBM when:** you have multiple years of area-level data and want to exploit the panel structure; your main model is GBM-based and you want spatial effects that integrate naturally with it; you have a large number of area-level covariates and want automatic variable selection; the GLM assumption of linear effects for area characteristics is visibly wrong.

**Stick with BYM2 when:** you are fitting a GLM and want spatial regularisation that is auditable for regulatory purposes; the adjacency-based prior is intuitive and defensible to a technical reviewer; fitting time is not a constraint and you have Stan infrastructure.

**Use Blier-Wong embeddings when:** you need to price territories with zero claims history; you have rich geographic feature data (census, land registry, road network) that you want to compress into a lower-dimensional spatial representation.

The paper does not claim to replace any of these. It fills a gap: a natural way to handle spatial panel structure within a GBM framework, with proper spatial cross-validation, for teams that are already committed to gradient boosting for their main model.

The Italian non-life application is convincing, but no UK-specific validation appears in the paper. UK home insurance has flood and subsidence as dominant drivers of spatial variation, both of which have distinct geographic patterns not obviously present in Italian property data. A UK team adopting this would need to validate it against postcode sector experience and compare to BYM2 on a held-out geography before changing production models.

---

**Paper:** Balzer, M. and Benlahlou, A. (2026). 'Gradient Boosting for Spatial Panel Models with Random and Fixed Effects.' arXiv:2603.14543. [https://arxiv.org/abs/2603.14543](https://arxiv.org/abs/2603.14543)

- [Geographic Ratemaking with Spatial Embeddings (Blier-Wong et al., ASTIN 2022)](/tags/#blier-wong) — spatial embeddings as an alternative approach to geographic pricing
- [BYM2 for Territory Ratemaking in UK Personal Lines](/tags/#BYM2) — the Bayesian CAR prior approach and when to prefer it
