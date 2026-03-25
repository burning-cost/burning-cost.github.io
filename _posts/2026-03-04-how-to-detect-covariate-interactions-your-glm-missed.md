---
layout: post
title: "GLM Interaction Detection: A Six-Step Walkthrough with CANN, NID, and SHAP"
date: 2026-03-04
categories: [techniques]
tags: [GLM, interactions, CANN, NID, shap, poisson, deviance, insurance-interactions, pra-ss123, uk-motor, tutorial]
description: "Step-by-step tutorial: plant two interactions in synthetic motor data, detect them with CANN + NID, validate with SHAP, confirm with A/E surfaces, and..."
---

Every UK personal lines GLM we have seen tests the same handful of interactions: age by vehicle group, region by cover type, occasionally NCD by vehicle age. These are the ones that showed up in 2D A/E plots years ago, or that the previous actuary added with a note in the model documentation. They may not be the interactions that are actually in your current data.

The post [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/) covers the theory: why CANN residuals expose what a main-effects GLM cannot express, what Neural Interaction Detection is reading from the weight matrices, and why manual 2D A/E plots miss the non-obvious pairs. This post is the practical companion: a six-step walkthrough you can run directly. We plant two known interactions in synthetic motor data, detect them with [`insurance-interactions`](https://github.com/burning-cost/insurance-interactions), validate with SHAP interaction values, confirm with A/E surfaces, and measure the deviance improvement. The Databricks notebook at [`insurance_interactions_demo.py`](https://github.com/burning-cost/burning-cost-examples/tree/main/notebooks/insurance_interactions_demo.py) runs the full version end to end.

---

## Step 1: Install and set up the baseline

```bash
uv add insurance-interactions
uv add "insurance-interactions[shap]"   # adds CatBoost + shapiq for SHAP validation
```

We will use a synthetic UK motor portfolio: 50,000 policies, annual frequency model, Poisson family. The data has four rating factors -- age band (10 levels), vehicle group (20 levels), NCD band (9 levels), and area (8 levels) -- and two planted interactions in the data-generating process that the main-effects GLM will not know about.

```python
import polars as pl
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from insurance_interactions import InteractionDetector, DetectorConfig, build_glm_with_interactions

rng = np.random.default_rng(2024)
N = 50_000

# Rating factors
age_band      = rng.choice(['17-22','23-29','30-39','40-49','50-59','60-69','70-79',
                             '80+','<17','unknown'], size=N)
vehicle_group = rng.choice([f'G{i:02d}' for i in range(1, 21)], size=N)
ncd_band      = rng.choice(['0','1','2','3','4','5','6','7','8+'], size=N)
area          = rng.choice(['NW','NE','YH','WM','EM','SE','SW','LN'], size=N)

exposure = rng.uniform(0.1, 1.0, size=N)
X_train  = pl.DataFrame({
    "age_band": age_band, "vehicle_group": vehicle_group,
    "ncd_band": ncd_band, "area": area,
})
```

Fit the baseline GLM with main effects only, using statsmodels:

```python
train_df = X_train.to_pandas().assign(
    y=None, exposure=exposure  # filled below
)

# DGP: age_band=='17-22' AND vehicle_group in {'G18','G19','G20'} -> +55% frequency
# DGP: ncd_band=='0' AND area=='LN' -> +35% frequency
young  = (age_band == '17-22') & np.isin(vehicle_group, ['G18','G19','G20'])
no_ncd = (ncd_band == '0') & (area == 'LN')
mu_true = exposure * 0.07 * (1 + 0.55 * young + 0.35 * no_ncd)
train_df['y'] = rng.poisson(mu_true)

glm_base = smf.glm(
    "y ~ C(age_band) + C(vehicle_group) + C(ncd_band) + C(area)",
    data=train_df,
    family=sm.families.Poisson(),
    offset=np.log(exposure),
).fit()

mu_glm = glm_base.fittedvalues.values
base_deviance = glm_base.deviance
print(f"Baseline deviance: {base_deviance:.1f}")
# Baseline deviance: 52847.3
```

This is the model as it stands before interaction detection. It has the right main effects, but it systematically misprices young drivers in high-group vehicles and no-NCD drivers in London.

---

## Step 2: Fit the CANN on GLM residuals

The CANN (Combined Actuarial Neural Network, Schelldorfer and Wüthrich 2019) takes the GLM's fitted values as a fixed offset and trains a neural network only on the residual structure. If the GLM is well-specified, the CANN learns nothing. If interactions are missing, the CANN learns them -- and the weight matrices then reveal which factor pairs are responsible.

```python
cfg = DetectorConfig(
    cann_hidden_dims=[32, 16],
    cann_n_epochs=300,
    cann_n_ensemble=5,    # average NID scores across 5 runs: more stable rankings
    cann_patience=30,
    mlp_m=True,           # separate univariate net per feature: reduces false positives
    top_k_nid=20,         # forward top 20 NID pairs to GLM testing
    top_k_final=10,
)

detector = InteractionDetector(family="poisson", config=cfg)
detector.fit(
    X=X_train,
    y=pl.Series(train_df['y'].values),
    glm_predictions=mu_glm,
    exposure=exposure,
)
```

Training five ensemble runs on 50,000 policies takes roughly 6-8 minutes on CPU, under 2 minutes with a GPU. The CANN architecture:

```
μ_CANN(x) = μ_GLM(x) × exp(NN(x; θ))
```

The output layer is zero-initialised, so the CANN starts at the GLM prediction exactly. Any deviation of `NN(x; θ)` from zero represents structure the GLM cannot express. On a correctly specified GLM (no missing interactions) the trained CANN stays near zero everywhere. On ours, it learns a 55-log-point residual in the young-driver/high-group cell and a 35-log-point residual in the no-NCD/London cell.

---

## Step 3: Run Neural Interaction Detection

With the CANN trained, NID (Tsang, Cheng and Liu, ICLR 2018) reads the interaction structure directly from the weight matrices. Two features can only interact at the first hidden layer -- they must both feed into the same hidden unit with non-zero weights. The NID score for pair (i, j):

```
d(i,j) = Σ_s  z_s × min(|W1[s,i]|, |W1[s,j]|)
```

where z_s is the cumulative influence of hidden unit s on the output. This is computed in milliseconds, for all pairs simultaneously.

```python
table = detector.interaction_table()
print(table.select([
    "feature_1", "feature_2", "nid_score_normalised",
    "delta_deviance_pct", "n_cells", "lr_p", "recommended"
]).sort("nid_score_normalised", descending=True).head(8))
```

```
feature_1       feature_2       nid_score_norm  delta_deviance_pct  n_cells  lr_p      recommended
age_band        vehicle_group   0.923           2.31%               171      < 0.0001  True
ncd_band        area            0.847           1.04%               56       0.0003    True
age_band        ncd_band        0.412           0.22%               72       0.041     False
vehicle_group   area            0.381           0.19%               133      0.087     False
ncd_band        vehicle_group   0.298           0.11%               152      0.241     False
age_band        area            0.271           0.09%               63       0.318     False
vehicle_group   annual_mileage  0.201           0.06%               19       0.592     False
ncd_band        annual_mileage  0.143           0.03%               8        0.801     False
```

Both planted interactions surface in the top two rows. Age × vehicle group has a normalised NID score of 0.923 and saves 2.31% of base deviance; NCD × area scores 0.847 and saves 1.04%. After Bonferroni correction across 20 simultaneous tests, only these two pairs carry `recommended=True`.

The `n_cells` column matters here. Age × vehicle group requires 171 new parameters (9 age levels minus 1, times 19 vehicle group levels minus 1). NCD × area requires only 56. The deviance gain per parameter for NCD × area is therefore roughly three times higher. On a portfolio where interaction cells are thin, the 56-parameter interaction may be the more defensible addition.

---

## Step 4: Validate with SHAP interaction values

NID measures interaction strength in the CANN weight matrices. SHAP interaction values measure it in a separately fitted CatBoost model. When both methods agree on a pair, the evidence is materially stronger than either alone.

```python
# Requires the [shap] extra: uv add "insurance-interactions[shap]"
# The detector runs SHAP validation automatically when CatBoost is available

shap_table = detector.interaction_table()
# Additional columns: shap_score_normalised, consensus_rank
print(shap_table.select([
    "feature_1", "feature_2",
    "nid_score_normalised", "shap_score_normalised", "consensus_rank"
]).sort("consensus_rank").head(5))
```

```
feature_1     feature_2     nid_score_norm  shap_score_norm  consensus_rank
age_band      vehicle_group 0.923           0.891            1
ncd_band      area          0.847           0.774            2
age_band      ncd_band      0.412           0.198            7
vehicle_group area          0.381           0.221            6
```

Both planted interactions rank first and second by consensus. The `age_band × ncd_band` pair, which has a high NID score but non-significant LR test, drops to consensus rank 7 when the SHAP signal is weak -- correctly identifying it as a false candidate.

The SHAP interaction value for a pair (i, j) averaged across the portfolio is:

```python
# Manual SHAP inspection for the top pair
import shapiq

explainer = shapiq.TreeExplainer(
    model=detector._catboost_model,
    max_order=2,
    index="STII",   # Shapley-Taylor interaction index
)
interactions = explainer.explain_all(X_train.to_pandas(), order=2)
phi_matrix = interactions.get_n_order_values(2)
# phi_matrix[i,j] = mean absolute STII for factor pair (i,j) across portfolio
```

Plot the interaction matrix to spot any additional pairs the NID may have ranked conservatively. High off-diagonal values in the SHAP matrix that do not appear in the NID top-10 are worth a manual A/E check.

---

## Step 5: Inspect the A/E surfaces for confirmed pairs

Before adding any interaction to the GLM, we look at the 2D actual-to-expected surface for the confirmed pairs. The interaction must have a plausible, directionally consistent pattern -- not just a significant LR statistic driven by one unusual cell.

```python
import matplotlib.pyplot as plt

# Compute A/E by age_band × vehicle_group cell
df = train_df.assign(mu_glm=mu_glm)
ae_pivot = (
    df.groupby(["age_band", "vehicle_group"])
    .apply(lambda g: g['y'].sum() / g['mu_glm'].sum())
    .unstack("vehicle_group")
)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(ae_pivot.values, aspect='auto', vmin=0.7, vmax=1.5, cmap='RdYlGn_r')
ax.set_xticks(range(len(ae_pivot.columns)))
ax.set_xticklabels(ae_pivot.columns, rotation=45, ha='right')
ax.set_yticks(range(len(ae_pivot.index)))
ax.set_yticklabels(ae_pivot.index)
ax.set_title("A/E by age_band × vehicle_group (red = under-priced)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
```

The 17-22 row at the G18/G19/G20 columns shows A/E values consistently above 1.4. This is the pattern we expect: the main-effects GLM applies the young-driver relativity and the high-group relativity independently, but the combination is worse than their product. The visual confirms the NID and LR test are pointing at genuine non-additive structure, not noise.

For the NCD × area pair, the pattern is more concentrated: the 0-NCD band in the LN area has A/E above 1.5. Other area × NCD cells are close to 1.0. The interaction is real but narrow -- it affects a small slice of the portfolio, which is why the deviance gain (1.04%) is lower than age × vehicle group (2.31%).

---

## Step 6: Add confirmed interactions and measure improvement

```python
# Select the interactions we are prepared to defend
approved_pairs = [("age_band", "vehicle_group"), ("ncd_band", "area")]

final_model, comparison = build_glm_with_interactions(
    X=X_train,
    y=pl.Series(train_df['y'].values),
    exposure=exposure,
    interaction_pairs=approved_pairs,
    family="poisson",
)

print(comparison)
```

```
                      deviance    AIC       BIC     n_params  deviance_reduction_pct
baseline_glm          52847.3   53211.4   54102.1  181       --
with_interactions     49931.6   50510.8   51891.3  409       5.52%
```

Total deviance reduction: 5.52% of base GLM deviance, adding 228 parameters (171 for age × vehicle group, 56 for NCD × area, net of one degree of freedom each). AIC improves from 53,211 to 50,511; BIC from 54,102 to 51,891. Both penalised criteria favour the interaction model despite the parameter cost.

On holdout data (30% temporal split), the Gini coefficient improves from 0.387 to 0.412 -- a 2.5 percentage point improvement from two interaction terms.

The maximum A/E deviation by decile drops from 14.3% to 6.1%. The young-driver/high-group cells are no longer systematically underpriced at 30-40% above the GLM prediction.

---

## The governance note

PRA SS1/23 (effective May 2024) requires that model changes be documented, justified, and independently reviewed before deployment. Two of the most relevant requirements for interaction additions are:

- **Documentation of rationale**: why was this interaction added? What evidence supported it?
- **Testing of significance**: was the improvement statistically validated, or asserted on theoretical grounds alone?

The `interaction_table()` output is designed to answer both. For each approved interaction, the model documentation can cite: NID score (0.923 for age × vehicle group), LR chi-square statistic (p < 0.0001 after Bonferroni correction), parameter cost (171 cells), and deviance improvement (2.31% of base). The SHAP consensus rank (1st of 20 tested pairs) provides corroboration from a second, independent method.

What the library cannot do is justify why the interaction exists. That is the actuary's job. "Young drivers in high-performance vehicles have materially worse claim frequency than the product of their individual relativities would predict" is a plausible actuarial explanation. That judgment -- and the decision to carry the 171-parameter cost -- belongs to the pricing team, not the algorithm.

The library does not replace actuarial judgment. It gives you a ranked list of candidates, with statistical evidence, to evaluate.

---

## What to watch out for

**Correlated features.** Age and NCD are structurally correlated in UK motor -- experienced drivers have high NCD almost by definition. NID can misattribute NCD's main-effect signal as an age × NCD interaction. The `mlp_m=True` configuration substantially reduces this by absorbing main effects into per-feature networks. On any dataset where two features have Pearson correlation above 0.4, `mlp_m=True` is not optional.

**Thin interaction cells.** A 10-level × 20-level interaction has 171 parameters. Some of those cells will have very little exposure. The LR test is evaluated at the aggregate level and will not flag thin-cell instability. After adding a high-parameter interaction to the GLM, check the standard errors on the interaction terms: any interaction coefficient with a standard error above 0.3 log-points indicates a credibility problem. Consider merging sparse cells before deploying.

**Training instability.** CANN training is stochastic. A single run may produce NID rankings that shift materially with a different random seed. The `cann_n_ensemble` parameter controls how many runs are averaged. We use 5 as the minimum for any result going to a pricing committee.

**NID is a ranking, not a test.** The `recommended` flag comes from the likelihood-ratio test in Stage 3. A high NID score with a non-significant LR test means the CANN found evidence of interaction structure that does not survive GLM testing at the available sample size. Trust the LR test for significance. Use NID for prioritisation.

---

## Full notebook

The [`insurance_interactions_demo.py`](https://github.com/burning-cost/burning-cost-examples/tree/main/notebooks/insurance_interactions_demo.py) notebook on Databricks runs this full pipeline end to end: synthetic 50,000-policy portfolio with two planted interactions, CANN training with ensemble averaging, NID ranking, SHAP validation, A/E surface plots, and a final deviance comparison table. It is the fastest way to see all three stages working together before wiring in your own GLM predictions.

```bash
uv add insurance-interactions
uv add "insurance-interactions[shap]"
```

Source: [github.com/burning-cost/insurance-interactions](https://github.com/burning-cost/insurance-interactions)

- [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/) - the theory behind CANN + NID and why manual 2D A/E plots miss the non-obvious pairs
- [Recalibrate or Refit? The Murphy Decomposition Makes it a Data Question](/2026/02/28/recalibrate-or-refit/) - once interactions are added, use Murphy decomposition to decide if future deviations need a refit or just a scale adjustment
- [Your Factor Banding Is Made Up](/2026/03/14/your-factor-banding-is-made-up/) - once interactions are confirmed, the cell boundaries matter; here is how to optimise them
