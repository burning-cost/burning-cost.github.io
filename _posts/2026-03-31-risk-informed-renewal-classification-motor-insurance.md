---
layout: post
title: "Risk-Informed Renewal Classification: Bridging the Pricing-Retention Silo in UK Motor"
date: 2026-03-31
categories: [techniques]
tags: [motor-insurance, renewal, retention, classification, XGBoost, SHAP, loading-ratio, PS21/5, ENBP, FCA, Consumer-Duty, python, insurance-optimise, NCB, churn]
description: "Boonkrong et al. (MDPI Risks 14(3):57, March 2026) show that adding actuarial pricing features to a renewal classifier materially improves prediction — the insight being that the retention model should see how much the renewal offer deviates from technical price. Here is how to adapt this for UK motor under PS21/5, why the 99.62% AUC is almost certainly an artefact, and what the implementation actually looks like."
author: burning-cost
---

UK motor insurers run two models at renewal and they largely do not talk to each other. The technical pricing model — a GLM or GBM built by actuaries on claims data — produces an expected loss cost and a technical premium. The retention model — usually a logistic regression or shallow GBM built by commercial analytics — produces a probability of lapsing given observed customer behaviour. These two outputs feed separately into a renewal pricing decision. The actuarial number sets the floor; the retention number informs the discount off the ceiling.

The information loss is structural. The retention model typically gets price change versus prior year, tenure, payment method, channel, and a handful of behavioural features. It does not get the ratio of the quoted premium to the technical price. It does not know whether the customer in front of it is being offered a price that is actuarially cheap or actuarially expensive. It models price sensitivity without controlling for whether the price is justified.

This is the silo problem. Boonkrong, Yang, Huang, and Simmachan (MDPI Risks 14(3):57, published 3 March 2026) are not the first to notice it, but they are among the first to demonstrate empirically — with real policy data — that feeding actuarial risk features directly into the renewal classifier produces a meaningfully better model. The mechanism is sound even if the reported numbers are almost certainly too good to be true.

---

## What Boonkrong et al. actually did

The paper uses 70,290 real-world Thai type-1 (comprehensive) private car policies from an unnamed Thai insurer. Class balance is close to even: 53.91% renewed (37,980 policies), 46.09% lapsed (32,468). They evaluate five classifiers — logistic regression, KNN, SVM, Random Forest, XGBoost — across multiple curated feature sets. The "risk-informed" framing means including actuarially meaningful features: net premium, sum insured (as a proxy for vehicle value), car age, and car group (the Thai insurer's underwriting risk segment).

The headline result is a 4-feature Random Forest achieving AUC = 99.62%, F1 = 98.15%. SHAP analysis confirms that net premium dominates the prediction — it accounts for the largest share of feature importance by a substantial margin.

Three feature selection methods were applied prior to model training: chi-square / mutual information filtering, stepwise logistic selection, and permutation importance. The reduced 4-feature set outperforms larger feature sets on held-out test data — a finding the paper attributes to the signal concentration in net premium and sum insured.

The Thai market has no equivalent of FCA PS21/5. There is no ENBP ceiling. Prices can move freely at renewal.

---

## Why 99.62% AUC is suspicious

A 4-feature classifier achieving near-oracle AUC on a balanced binary classification task should immediately raise a flag. The paper does not discuss feature leakage explicitly, but the mechanism is straightforward: **net premium at renewal partially encodes the insurer's own pricing decision**.

When an insurer quotes a high renewal premium, they are implicitly signalling their assessment of the risk — and their estimate of the customer's price sensitivity. Customers who receive very high renewal quotes are more likely to lapse; the insurer often knows this when they set the price, and the price reflects it. The feature "net premium at renewal" therefore contains both a risk signal (expensive risks get higher premiums) and a pricing-intent signal (the insurer already modelled lapse probability to set that premium). The retention model then recovers the insurer's prior belief about lapse, dressed up as a learned feature.

This is a partially circular dependency. The classifier is not learning to predict renewal from first principles — it is partially recovering the insurer's own pricing model. In a real deployment, where you are trying to decide *what* to quote, you cannot use the final quoted premium as an input to the model that informs that quote. You need the technical price, not the quoted price.

In practice, expect AUC in the range of 0.70–0.80 on a properly constructed out-of-time UK test set where features are computed at the point of pricing, not after. The 99.62% figure is an interesting empirical footnote, not a deployment benchmark.

The core insight — that risk-informed features improve retention models — survives this critique. The 99.62% does not need to be real for the direction to be right.

---

## The feature that matters: loading ratio

Strip away the circularity and the useful residue is this: the retention model should know whether the customer is being offered a price that is actuarially expensive or cheap relative to their risk. Not absolute price, and not price change versus prior year — both of which confound risk changes with pricing decisions — but the ratio of quoted premium to technical price.

We call this the loading ratio:

$$\text{loading\_ratio} = \frac{\text{quoted\_renewal\_premium}}{\text{technical\_price}}$$

A loading ratio of 1.0 means the customer is being quoted at exactly their expected loss cost (plus whatever fixed loadings are in the technical price definition). A ratio of 1.4 means they are being quoted 40% above technical price. A ratio of 0.9 means they are being quoted below technical — which in UK motor post-PS21/5 is common for competitive segments where the technical price is conservative.

The mechanism linking loading ratio to renewal probability is causal and clear. Higher loading ratio means the customer is more likely to find a cheaper quote at market, because competitors who price more efficiently will undercut. Lower loading ratio means the customer is cheap relative to their risk and is less likely to find a better deal elsewhere — but also means the insurer is accepting thin or negative margin to retain them.

When the retention model sees this feature, it is no longer modelling price sensitivity in the abstract. It is modelling price sensitivity conditional on where the customer sits relative to their actuarial fair value. That is the right question.

---

## UK translation: PS21/5, ENBP, and NCB

The UK version of this problem has regulatory structure that actually helps.

FCA PS21/5 (ICOBS 6B, effective 1 January 2022) bans motor and home insurers from quoting a renewal price higher than the equivalent new business price (ENBP). This bounds the loading ratio from above in a specific way: the quoted renewal premium cannot exceed what the insurer would offer the same customer as a new business prospect. You cannot walk prices up at renewal beyond ENBP.

This constraint changes the shape of the problem. In the Thai data, net premium at renewal varies freely and encodes pricing intent without bound. In UK motor, the upper bound on the loading ratio is determined by ENBP, which is itself typically close to technical price plus new business expenses. So the useful derived feature is not just loading ratio but also how close the customer is to the ENBP ceiling:

$$\text{enbp\_proximity} = \frac{\text{quoted\_renewal\_premium}}{\text{enbp\_price}}$$

A value close to 1.0 means the insurer has already offered the maximum permissible premium. There is no room to reprice upward. A value of 0.85 means there is headroom. This feature encodes both competitive position (relative to the insurer's own new business rate) and regulatory constraint.

FCA EP25/2 (July 2025), which reviewed the impact of GIPP remedies, confirmed £1.6bn in motor consumer savings but noted that opaque techniques within ENBP constraints remain a residual concern. A renewal classifier stratified by loading ratio directly surfaces whether certain risk profiles are being priced to the ENBP ceiling while others are retained at below-technical prices. That is both a Consumer Duty fair value question and a commercially useful segmentation.

Beyond the loading structure, UK motor has two features absent from the Thai paper that any actuary would insist on including:

- **NCB years**: No Claims Bonus level is a direct proxy for claims history over the tenure of the policy. It encodes risk quality in a way that net premium alone cannot.
- **Claims in prior year**: A binary (or count) flag for claims in the prior policy year. A customer who has claimed is facing a premium increase regardless of loading ratio, and the renewal probability distribution is sharply different from a claim-free customer.

---

## Python implementation

We have added `RiskInformedRetentionModel` to `insurance-optimise` as an extension to the existing demand subpackage. It handles the feature engineering — computing loading ratio, enbp proximity, and related derived features — and wraps a configurable classifier (CatBoost by default, GBT or logistic as alternatives).

```python
from insurance_optimise.demand import RiskInformedRetentionModel

model = RiskInformedRetentionModel(
    model_type='catboost',
    technical_price_col='technical_price',
    renewal_price_col='renewal_price',
    enbp_price_col='enbp_price',
    feature_cols=['tenure_years', 'payment_method', 'channel'],
)
model.fit(renewals_df)
lapse_prob = model.predict_proba(renewals_df)
```

The model computes `loading_ratio` and `enbp_proximity` internally from the three price columns. The `feature_cols` list carries the behavioural and demographic features that sit alongside the risk-informed derived features — you do not include the raw price columns there. NCB and prior claims are automatically included if present as `ncb_years` and `claims_prior_year` in the dataframe; the constructor has explicit parameters to override these column names.

The `predict_proba` output is a lapse probability series indexed to `renewals_df`, ready to feed directly into `insurance-optimise`'s renewal pricing optimiser. The standard pipeline is:

1. Score renewals with `RiskInformedRetentionModel` to get per-policy lapse probability.
2. Feed lapse probability + expected loss cost + ENBP constraint into the renewal price optimiser.
3. Optimiser finds the premium that maximises expected contribution subject to the ENBP ceiling and a fair value floor.

The SHAP interface follows the same pattern as the rest of the `insurance-optimise` explainability stack:

```python
import shap

explainer = shap.TreeExplainer(model.classifier_)
shap_values = explainer.shap_values(model.transform(renewals_df))
```

`model.transform()` returns the engineered feature matrix including loading ratio and enbp proximity, so the SHAP output directly attributes prediction importance to these derived features rather than the raw price columns. In our testing on synthetic UK-style data, loading ratio typically ranks first or second in SHAP importance — which is the expected result if the feature is doing what it should.

---

## Caveats and validation

The paper's 99.62% AUC establishes an upper bound on what is achievable when features contain the insurer's own pricing intent. In production, where you need to use pre-quote features, the relevant benchmark is the out-of-time AUC — and that means validating on a test set from a different calendar quarter than training, not just a random 30% holdout.

We expect 0.70–0.80 AUC on a properly constructed UK out-of-time test set. An AUC of 0.75 on held-out renewal decisions is a useful classifier. It is not 99.62%, but it does not need to be.

Two additional validation steps matter for UK deployment:

**Segment calibration.** Lapse probabilities should be calibrated per channel (PCW versus direct versus broker) and per tenure band. PCW customers have materially higher price sensitivity and a different baseline lapse rate. A model that pools these segments will be miscalibrated on the tails.

**Loading ratio monotonicity check.** After fitting, verify that the partial dependence of lapse probability on loading ratio is monotonically increasing. If it is not — if the model shows that customers at high loading ratios are predicted to renew at higher rates than customers at moderate loading ratios — this is a sign of either collinearity with another feature or overfitting in that region of the input space. The relationship should be monotone and that is a model assumption worth enforcing explicitly if the data does not produce it naturally.

---

## Where to find the code

`RiskInformedRetentionModel` is in the `insurance-optimise` repository at [github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise), under `insurance_optimise/demand/risk_informed.py`. The extension is approximately 400 lines including feature engineering and SHAP utilities. The `examples/` directory has a worked notebook on synthetic UK motor renewal data.

The Boonkrong et al. paper is open access: DOI [10.3390/risks14030057](https://doi.org/10.3390/risks14030057).
