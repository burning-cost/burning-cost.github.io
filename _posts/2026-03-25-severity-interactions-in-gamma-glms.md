---
layout: post
title: "Severity Interactions in Gamma GLMs: Weaker Signal, Higher Bar"
date: 2026-03-25
categories: [techniques, pricing]
tags: [GLM, interactions, CANN, NID, gamma, severity, poisson, insurance-interactions, uk-motor, tutorial, python]
description: "Applying CANN + NID to severity (Gamma) GLMs. Why the signal is weaker than frequency, what configuration changes are needed, and when a severity interaction is worth adding."
---

Most UK personal lines teams, if they have tried automated interaction detection at all, have run it on the frequency model. The Poisson GLM. Claim counts, exposure-weighted, everything clean and Gaussian-ish in the residuals. That is the right place to start. Frequency interactions are easier to find, easier to confirm, and easier to defend.

Severity is harder. The Gamma GLM for claim amounts is noisier by construction — a single large loss can shift A/E surfaces for an entire cell — and the interactions that exist are rarer and smaller. When we run the [`insurance-interactions`](/insurance-gam/) CANN + NID pipeline on a severity model, we do not get the same clean separation between genuine interactions and noise that we get on frequency. The top NID pair has a normalised score of 0.7 instead of 0.92. Two of the top five pairs fail the likelihood-ratio test. The deviance gains are measured in 0.3%, not 2.3%.

That does not mean severity interactions are not worth looking for. It means the bar is higher, the configuration needs adjusting, and the process of confirming what the algorithm flags requires more careful actuarial judgement.

This post covers the mechanics of running the detector on a Gamma GLM, the configuration choices that matter for noisy targets, and the criteria we use to decide whether a flagged severity interaction is genuinely worth adding.

---

## Why Gamma residuals are harder to learn from

The CANN works by learning whatever the main-effects GLM cannot express. In a correctly specified model, the CANN learns nothing. In a frequency model with a missing interaction, the CANN sees a systematic pattern in the Poisson residuals — the cells with the interaction are consistently underpredicted by a factor of 1.3 or 1.5, and that pattern is stable across the training data. A [32, 16] MLP trained on 50,000 policies picks it up reliably within a few hundred epochs.

Claim amounts are different. The coefficient of variation for an individual severity observation is high — often above 1.5 for UK motor — because a £500 repair and a £15,000 write-off sit in the same cell. The interaction signal, when it exists, is a shift in the conditional mean claim amount for a particular factor combination. That shift is real, but it is buried under the variance of individual observations in a way that frequency interactions are not.

The practical consequence: a severity interaction that produces a 0.4% deviance gain requires the CANN to distinguish a 4-5% shift in conditional mean from an ambient noise level that can easily be 30-40%. On frequency, the equivalent interaction might produce a 15-20% lift in a specific cell, which is clearly visible above noise.

This is not a bug in the method. It is a property of the data. The configuration changes below compensate.

---

## Configuration for severity

```bash
uv add insurance-interactions
```

The default `DetectorConfig` is calibrated for frequency. For severity, three parameters need changing:

```python
from insurance_interactions import InteractionDetector, DetectorConfig

cfg = DetectorConfig(
    cann_n_ensemble=7,          # more runs needed for stable NID rankings
    cann_n_epochs=400,          # longer training: severity loss surface is flatter
    cann_patience=40,           # more patience: early stopping fires too soon on noisy targets
    cann_hidden_dims=[64, 32],  # wider net: severity has more variance to absorb
    mlp_m=True,                 # always True for severity — correlation false positives are worse
    top_k_nid=15,               # test fewer pairs: LR tests are more expensive on severity data
    alpha_bonferroni=0.05,
)

sev_detector = InteractionDetector(family="gamma", config=cfg)
```

The key changes:

**`cann_n_ensemble=7`**. NID rankings on severity data are more sensitive to training randomness than on frequency. A single run can produce a top-5 that shifts materially on the next seed. Seven ensemble runs average out enough stochasticity that the top 3-4 pairs are usually stable.

**`cann_n_epochs=400, cann_patience=40`**. The Gamma deviance loss surface is flatter than Poisson. The CANN's correction stays closer to zero for longer before it finds genuine residual structure. With the default `n_epochs=200` and `patience=20`, early stopping regularly terminates training before the network has properly fitted the interaction signal. Extend both.

**`cann_hidden_dims=[64, 32]`**. The default `[32, 16]` architecture has 512 + 256 = 768 parameters in the hidden layers. On a severity model with high per-observation variance, a wider network has more capacity to separate the interaction signal from noise. Going beyond `[64, 32]` rarely helps and starts producing overfitting artefacts.

**`mlp_m=True`**. For severity, this is even more important than for frequency. If the mean severity has a strong main effect on one factor — vehicle group, for instance, where prestige cars have systematically higher repair costs — NID can misattribute that main-effect variance as an interaction with adjacent factors. MLP-M forces each feature's main effect into its own small network, leaving the interaction MLP clean.

---

## Fitting on severity data

Severity models are fitted on claims-only data. The exposure weight for the Gamma GLM is claim count (the number of claims that produced the observed amounts), not policy years.

```python
import polars as pl
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from insurance_interactions import InteractionDetector, DetectorConfig, build_glm_with_interactions

# X_claims: DataFrame of rating factors, one row per claim
# y_amounts: observed claim amounts
# claim_counts: used as weights (often 1.0 per row if X_claims is claim-level)

glm_sev = smf.glm(
    "amount ~ C(age_band) + C(vehicle_group) + C(ncd_band) + C(area)",
    data=claims_df,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    freq_weights=claims_df["claim_count"],   # weight by claim count if aggregated
).fit()

mu_sev_glm = glm_sev.fittedvalues.values
```

Pass the Gamma GLM's fitted values as `glm_predictions`. These enter the CANN as a log-space offset, exactly as in the frequency case:

```python
sev_detector.fit(
    X=X_claims,
    y=y_amounts,
    glm_predictions=mu_sev_glm,
    exposure=claim_counts,   # claim count as exposure weight
)
```

If your data is claim-level (one row per claim, not aggregated), set `exposure=None` — the library defaults to all-ones weights, which is correct for claim-level Gamma data.

---

## Reading the severity interaction table

```python
table = sev_detector.interaction_table()
print(table.select([
    "feature_1", "feature_2", "nid_score_normalised",
    "delta_deviance_pct", "n_cells", "lr_p", "recommended"
]).sort("nid_score_normalised", descending=True).head(8))
```

A typical output on a real UK motor severity portfolio with 40,000 claims:

```
feature_1       feature_2       nid_score_norm  delta_deviance_pct  n_cells  lr_p      recommended
vehicle_group   age_band        0.741           0.38%               171      0.003     True
cover_type      area            0.619           0.29%               35       0.011     True
ncd_band        vehicle_group   0.531           0.17%               152      0.062     False
age_band        area            0.489           0.14%               63       0.091     False
cover_type      vehicle_group   0.441           0.11%               95       0.183     False
ncd_band        area            0.388           0.08%               56       0.312     False
age_band        ncd_band        0.312           0.05%               72       0.541     False
cover_type      age_band        0.271           0.03%               45       0.719     False
```

Compare this to the frequency output from the same portfolio:

- Normalised NID scores top out at 0.74 vs 0.92. The CANN has found something, but the interaction structure is less sharply defined.
- The top two pairs are significant after Bonferroni correction, but at p=0.003 and p=0.011 rather than p<0.0001. Both would survive a 0.05/15 = 0.003 threshold, but only just.
- Six of eight pairs fail the LR test. On frequency, the same model had only four of eight fail.
- Deviance gains are 0.38% and 0.29%, against 2.31% and 1.04% on frequency from the same factors.

The `n_cells` column is now more important than ever. Vehicle group × age band at 171 parameters saving 0.38% of base deviance gives a deviance-per-parameter ratio of 0.38/171 = 0.0022%. Cover type × area at 35 parameters saving 0.29% gives 0.29/35 = 0.0083% — nearly four times more efficient. If thin-cell credibility is a concern, cover type × area is the better addition despite the lower absolute deviance gain.

---

## The higher confirmation bar

On a frequency model, a pair with `recommended=True` and a well-separated A/E surface goes into the GLM. On severity, we require two additional checks before adding any interaction.

**Check 1: cell-level credibility.** The LR test is computed at portfolio level. A 0.38% deviance gain with p=0.003 can come from a few high-exposure cells with consistent A/E above 1.3, or it can come from many sparse cells with noisy A/E. The difference matters for deployment. After running the LR test, count how many interaction cells have fewer than 100 claims:

```python
# Manual check for vehicle_group x age_band
cell_counts = (
    claims_df
    .groupby(["vehicle_group", "age_band"])
    .agg(pl.count().alias("n_claims"))
)
sparse_cells = cell_counts.filter(pl.col("n_claims") < 100).shape[0]
total_cells = cell_counts.shape[0]
print(f"Cells with < 100 claims: {sparse_cells} of {total_cells}")
```

If more than 40% of cells have fewer than 100 claims, the interaction is credibility-thin. The statistical test says it is real; the data volume says it will be unstable in deployment. Consider collapsing vehicle groups before adding the interaction.

**Check 2: direction consistency.** Compute the A/E surface for severity, not frequency. Severity A/E should have a coherent directional pattern — a clear region where the combination produces higher average claim amounts than the product of the two main effects would predict. Dispersed A/E deviations without a directional story are usually noise lifted to significance by a large portfolio.

```python
sev_ae = (
    claims_df
    .groupby(["vehicle_group", "age_band"])
    .agg([
        pl.col("amount").sum().alias("actual"),
        pl.col("mu_sev_glm").sum().alias("expected"),
        pl.count().alias("n_claims"),
    ])
    .with_columns((pl.col("actual") / pl.col("expected")).alias("ae"))
    .filter(pl.col("n_claims") >= 50)  # only cells with credible exposure
)
```

Plot this as a heatmap. For a genuine severity interaction, you expect a connected region of high A/E — not a random scatter. If vehicle groups 14-20 combined with age bands 17-29 consistently show A/E above 1.25, that is the story. If A/E above 1.25 appears in isolated, non-adjacent cells, it is noise.

---

## When severity interactions are worth adding

We have added severity interactions to about a dozen UK motor and home models over the past few years. The honest summary:

**Frequency interactions are present in almost all models.** Run the detection on any reasonably large UK motor frequency GLM and you will find at least one pair with a deviance gain above 0.5% and a Bonferroni-significant LR test. The young-driver/high-group interaction in particular is nearly universal.

**Severity interactions are present in roughly half.** Of the severity models we have analysed with CANN + NID, about half produced at least one pair that survived our three-stage confirmation process (NID ranking, LR test, credibility and direction check). The other half produced no severity interaction worth adding.

**The useful severity interactions tend to involve cover type.** Comprehensive versus TPO versus TPFT is a stronger moderator of severity than frequency. A no-NCD driver on comprehensive cover in a high vehicle group has higher average claim costs than a multiplicative model would predict because their comprehensive claims include a higher proportion of at-fault-own-vehicle costs. That is a real effect and it is detectable.

**Vehicle age × vehicle group interactions are real but expensive.** A 15-year-old vehicle in a high vehicle group has different severity characteristics than the products of the two main effects would suggest — older prestige cars are repaired at higher hourly rates by independent garages rather than approved repairers. This interaction is real, it shows up consistently in NID, and the parameter cost (vehicle age deciles × vehicle groups = 9 × 19 = 171 cells on a typical banding) is large. We have added it twice; both times it improved holdout Gini by 0.8-1.2 percentage points on severity.

**NCD × area interactions are frequency effects, not severity.** If this pair surfaces in a severity run, it is usually a false positive. The no-NCD/London cell has elevated claim frequency (young drivers, high-density road usage); severity for those claims is not materially different. NID sometimes picks this up as a severity interaction because the cell has elevated claim counts, which affects the GLM residuals in ways that look like an interaction to the CANN. Always check whether a severity interaction candidate has an obvious frequency explanation before adding it.

---

## Running frequency and severity together

The library is designed to run separately for each component. The complete workflow for a two-part frequency-severity model:

```python
# Frequency: Poisson GLM on all policies
freq_cfg = DetectorConfig(cann_n_ensemble=5, mlp_m=True)
freq_detector = InteractionDetector(family="poisson", config=freq_cfg)
freq_detector.fit(
    X=X_all,
    y=claim_counts,
    glm_predictions=mu_freq_glm,
    exposure=exposure,
)

freq_pairs = freq_detector.suggest_interactions(top_k=5)
# e.g. [("age_band", "vehicle_group"), ("ncd_band", "area")]

# Severity: Gamma GLM on claims only
sev_cfg = DetectorConfig(
    cann_n_ensemble=7,
    cann_n_epochs=400,
    cann_patience=40,
    cann_hidden_dims=[64, 32],
    mlp_m=True,
    top_k_nid=15,
)
sev_detector = InteractionDetector(family="gamma", config=sev_cfg)
sev_detector.fit(
    X=X_claims,
    y=claim_amounts,
    glm_predictions=mu_sev_glm,
    exposure=None,  # claim-level data, all weights = 1
)

sev_pairs = sev_detector.suggest_interactions(top_k=3, require_significant=True)
# e.g. [("vehicle_group", "age_band"), ("cover_type", "area")]
```

Then add the confirmed pairs to each model independently:

```python
freq_model, freq_comparison = build_glm_with_interactions(
    X=X_all,
    y=claim_counts,
    exposure=exposure,
    interaction_pairs=freq_pairs,
    family="poisson",
)

sev_model, sev_comparison = build_glm_with_interactions(
    X=X_claims,
    y=claim_amounts,
    exposure=None,
    interaction_pairs=sev_pairs,
    family="gamma",
)
```

It is normal for the lists to overlap partially. If age × vehicle group appears in both `freq_pairs` and `sev_pairs`, the interaction has evidence in both model components. That is the strongest possible case for including it. If it only appears in frequency, the interaction is a frequency effect and should not be added to the severity GLM based on the frequency evidence alone.

---

## Limitations specific to severity

**Sample size matters more.** The CANN's Gamma deviance loss has a much flatter gradient surface than Poisson on sparse data. Below roughly 15,000 claims, NID rankings become unreliable for severity. The LR tests in `test_interactions()` still work below this — they are standard GLM statistics and do not depend on the CANN — but using the NID ranking to prioritise which pairs to test requires a larger claim base than the frequency equivalent. The workaround: increase `top_k_nid` to 20 and test more pairs with the LR, using the GLM test statistics rather than the NID ranking for prioritisation.

**Large losses distort the LR test.** A single £500,000 claim in a cell with 200 ordinary claims can shift the cell's deviance by more than the interaction signal you are trying to detect. Before running interaction detection on a severity model, either: winsorise at the 99th percentile (£50,000-80,000 for UK motor); or run a large loss loading analysis separately and remove large losses above a threshold from the severity GLM. Adding the large loss separately as a function of risk factors and putting only attritional claims through the GLM interaction detection is cleaner.

**The CANN trains on per-claim data, not exposure.** For claim-level severity data, there is no meaningful exposure to pass. The `exposure` parameter in `CANN.fit()` defaults to ones when `None`. This is correct: you are modelling claim severity conditional on a claim occurring, and each row represents one claim. Do not pass policy exposure here — it has no meaning in the severity model and will distort the deviance loss weighting.

---

## What the frequency post did not say

The two posts on frequency interaction detection ([Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/), [GLM Interaction Detection: A Six-Step Walkthrough](/2026/03/04/how-to-detect-covariate-interactions-your-glm-missed/)) were honest about frequency limitations. Neither said anything about severity because severity is genuinely harder and we had not worked through the configuration differences carefully enough to write about it.

The short version: run it, use more ensemble runs and more epochs, require both NID and LR confirmation, check credibility before deployment, and expect roughly half your runs to produce no severity interaction worth adding. That is not failure — it is the method correctly reporting that the data does not contain the effect you were looking for.

```bash
uv add insurance-interactions
```

Source: [github.com/burning-cost/insurance-interactions](https://github.com/burning-cost/insurance-interactions)

- [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/) — frequency interaction detection: theory and three-stage pipeline
- [GLM Interaction Detection: A Six-Step Walkthrough](/2026/03/04/how-to-detect-covariate-interactions-your-glm-missed/) — the practical frequency walkthrough with planted interactions
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — if severity interactions improve your Gamma model, the next step is modelling the dispersion too
