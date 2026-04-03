---
layout: post
title: "Five Ways Insurance Fraud ML Papers Mislead You (And How to Spot Them)"
date: 2026-04-03
categories: [techniques]
tags: [fraud-detection, focal-loss, XGBoost, AUC-PR, AUC-ROC, class-imbalance, model-validation, temporal-validation, SMOTE, Boabang-Gyamerah-2025, arXiv-2508.02283, Consumer-Duty, UK-GDPR, Failure-to-Prevent-Fraud, motor-insurance, benchmarking, cost-sensitive-learning]
description: "Insurance fraud ML papers routinely overstate their results through five avoidable errors: wrong evaluation metric, no external baseline, random train/test split, foreign data, missing cost matrix. The Boabang three-stage focal loss paper (AUC=0.63) is a textbook illustration of all five."
author: burning-cost
---

Insurance fraud is a legitimate modelling problem with a genuine business case. The ABI confirmed £1.16bn in detected fraud in 2024 — 98,400 claims, up 12% — and motor fraud alone accounts for £576m of that. The industry has real need for better detection tooling.

The academic literature on insurance fraud ML is less useful than it should be. A steady stream of papers announces new approaches — focal loss, curriculum training, contrastive learning, hybrid ensemble architectures — each claiming to outperform what came before. Most of them cannot be used to make any defensible conclusion about what actually works. Not because the techniques are wrong, but because the experimental designs share the same five errors, repeated paper after paper.

A new arXiv paper from Boabang and Gyamerah (2508.02283, August 2025), introducing a three-stage focal loss curriculum for tabular insurance fraud, illustrates all five cleanly. We use it as a worked example. This is not a review of that paper specifically — we covered that [in March](/2026/03/31/focal-loss-insurance-fraud-detection/) — but a practitioner's checklist for reading the literature without being misled.

---

## Error 1: Reporting AUC-ROC instead of AUC-PR

The Receiver Operating Characteristic curve plots true positive rate against false positive rate across all thresholds. It is a sensible metric for balanced classification problems. For fraud detection at 2.15% prevalence, it is almost useless — because the false positive rate denominator is dominated by the overwhelming majority of genuine claims.

Consider: a model that scores the top 10% of claims as high-risk and correctly captures 80% of all fraud cases has a true positive rate of 0.80. Its false positive rate is (0.10 × N − 0.80 × fraud_N) / genuine_N. At 2.15% fraud prevalence with 2.4 million motor claims: about 51,600 fraud cases and 2,348,400 genuine cases. The model flags 240,000 claims. Of those, 41,280 are fraud (80% recall). The false positive rate is 198,720 / 2,348,400 = 0.085. An AUC-ROC that includes this model might easily come in at 0.90 — a number that sounds impressive, but the model is flagging 198,720 genuine claims for every 41,280 correctly identified fraud cases. The precision is 17%.

The Precision-Recall curve does not have this pathology. It plots precision against recall directly. A model that flags 240,000 claims and gets 17% precision will show that 17% on the PR curve. The AUC-PR for UK motor fraud models, trained on detected labels only, typically sits between 0.35 and 0.60 for reasonable models. That range is honest about the difficulty of the problem.

The Boabang paper reports AUC-ROC = 0.63. At 2% prevalence, a naive model that ranks genuinely ambiguous claims highly could achieve this with modest real skill. The AUC-PR for the same model is not reported. Without it, the result cannot be evaluated.

**What to check when reading:** Does the paper report AUC-PR, F2-score (which weights recall over precision, appropriate when false negatives are costlier), or at minimum precision@recall curves? If only AUC-ROC is reported on an imbalanced fraud dataset, the key performance figure is missing.

---

## Error 2: No external baseline

The Boabang three-stage curriculum approach delivers AUC-ROC = 0.630 and F1 = 0.415. Against what? Against three ablated versions of the same method: convex-only (AUC 0.532), non-convex with alpha=0.25 (AUC 0.586), non-convex with alpha=0.50 (AUC 0.580). The improvement over convex-only is +18.5% on AUC-ROC. That is real: the curriculum training helps with convergence.

What the paper does not include: XGBoost with `scale_pos_weight`, LightGBM, logistic regression with class weighting, SMOTE + Random Forest, or a cost-sensitive SVM. Without these, it is impossible to answer the only question that matters to a practitioner: *is this better than what I would already use?*

This is not an oversight. It is a structural feature of the incentive landscape: a paper that proposes method X and shows X beating ablations of X will always find a journal home. A paper that proposes X and shows it losing to XGBoost does not get submitted.

The practical consequence is that the fraud ML literature is almost entirely composed of papers that demonstrate internal improvements over their own ablations. The inter-method comparison — the one that would tell you whether to adopt any of these approaches — is systematically missing. Banulescu-Radu and Yankol-Schalck (Journal of Risk and Insurance, 2024, ARIA Prize winner) is the nearest exception we know of, specifically because it uses XGBoost as the explicit benchmark on a real household insurance dataset. It is also the paper the fraud literature tends not to cite.

**What to check when reading:** Are the baselines external methods that a practitioner would actually use, or only ablations of the proposed method? If the comparison set is entirely within the paper's own framework, the result tells you only that the method is better than a worse version of itself.

---

## Error 3: Random train/test split

Insurance fraud is a non-stationary phenomenon. Fraud rings evolve. Application fraud patterns shift as insurers tighten controls. Staged accident networks move between regions as enforcement activity follows them. A model trained on 2022-2024 fraud patterns will degrade against 2025 fraud patterns in ways that a random 80/20 train/test split will not reveal, because both folds contain the same temporal fraud patterns.

Every paper we reviewed — including Boabang, the CP-SMOTE motor insurance paper (Journal of Big Data, 2025), and the PMC credit card focal loss study (2025) — uses random splitting. This systematically overstates production performance. The realistic estimate is that temporal validation will show 10-20% AUC degradation versus random split for a model that was tested on held-out data from the same time period as training.

For a model that already achieves AUC-ROC = 0.63 under random split, the temporal validation result could easily be 0.53 — near random. That is the result that matters for a practitioner deciding whether to deploy.

**What to check when reading:** Does the paper use a temporal train/test split (train on claims from years t-1, test on year t) or a random split? For any production fraud model, temporal validation is the relevant test. Random split performance is an upper bound, not an estimate.

---

## Error 4: Foreign data with unknown prevalence

The Boabang dataset is US automobile insurance claims from a single unnamed large US insurer, N=39,981 records. The fraud prevalence rate is not disclosed.

This matters in two ways. First, the 1:46 imbalance we have calculated for UK motor (2.15% detected prevalence from ABI figures against 2.4 million motor claims in 2024) may not apply. US auto insurance fraud rates vary widely by state — California and Florida run significantly higher detected rates than the national average. If the unnamed US insurer had a 5% or 8% fraud prevalence in its dataset, the imbalance problem is materially less severe than UK motor, and the case for focal loss's modulating factor is correspondingly weaker or stronger.

Second, the feature set reflects US insurance product structures. The top SHAP features in the Boabang paper include `policy_state` (which US state the policy was written in), `insured_education_level`, and `insured_sex`. Education level is not a standard UK motor rating factor. Sex has been prohibited as a pricing factor since the *Test-Achats* ruling (2012) and using it as a fraud predictor in the UK raises Equality Act 2010 questions (indirect discrimination via a proxy for protected characteristics). The feature list tells you the model was built for a different regulatory and product context than UK personal lines.

A model trained on this data will not transfer to a UK motor portfolio without retraining from scratch on UK data, with UK features, against UK fraud labels.

**What to check when reading:** Is the dataset origin and fraud prevalence disclosed? Are the features used ones that would be available and legally permissible in UK insurance? Papers trained on Kaggle credit card datasets (US banking, 0.17% prevalence) and evaluated on AUC-ROC will, essentially always, look impressive and will tell you nothing about UK motor fraud detection.

---

## Error 5: Missing cost matrix

Every metric in the Boabang paper — AUC-ROC, F1, accuracy, precision, recall — treats a false positive and a false negative as symmetric. They are not.

In UK personal lines insurance, a false positive from a fraud model (genuine claim incorrectly flagged as suspicious, leading to investigation delay or denial) has the following potential costs:

- **FCA Consumer Duty (PS22/9):** Claim handling must achieve good outcomes for customers. An incorrectly declined claim is a bad outcome. The firm must be able to demonstrate that its fraud triage process does not systematically disadvantage particular customer groups.
- **Financial Ombudsman Service:** FOS referrals cost the firm £750 in case fees, and if the FOS finds in the customer's favour, the firm pays the claim plus potential compensation. The FOS consistently upholds a substantial proportion of claims-related complaints — the reputational and financial cost of systematic false positives is material.
- **UK GDPR Article 22:** Where the fraud score triggers an automated significant decision affecting the customer — claim denial or enhanced investigation — the customer has the right to human review and a meaningful explanation. A model whose explanations are not reliable (SHAP instability, as noted in our March post) cannot satisfy this right.
- **Failure to Prevent Fraud Act (in force 1 September 2025):** For large insurers, the FTPF offence works in the opposite direction — it increases the regulatory pressure to deploy systematic fraud detection. But it also sharpens the focus on the quality of the detection: flagging genuine customers incorrectly is not a "reasonable procedure" defence; it is a liability.

A false negative (undetected fraud claim paid out) costs the insurer the claim value. For UK motor, average paid fraud claim is approximately £576m / 51,700 = £11,100. A false positive that results in FOS referral and upheld complaint costs the insurer the claim plus FOS fees plus potential compensation — in some cases, more than simply paying the claim.

The F-beta metric, with beta > 1 to weight recall over precision, is closer to the right objective function for fraud detection. Most papers use F1 (beta=1, equal weighting). The right beta for UK motor fraud depends on the cost ratio of false negatives to false positives — which varies by product line, claim type, and current FOS complaint rate. No paper in this literature attempts to estimate it.

**What to check when reading:** Does the paper optimise a metric that reflects the actual costs of the two error types in the jurisdiction and product context you care about? F1 on US auto insurance is not F2 on UK motor. If the cost matrix is absent, the paper's metric optimisation has an unknown relationship to the business objective you actually have.

---

## What a useful benchmark paper would look like

No clean head-to-head comparison of focal loss, XGBoost with `scale_pos_weight`, SMOTENC + XGBoost, and cost-sensitive LightGBM exists on real insurance claims data with matched hyperparameter tuning, temporal splitting, and AUC-PR as the primary metric.

Such a paper would need:

- A single dataset with disclosed fraud prevalence and temporal ordering (ideally a public one, since proprietary data cannot be reproduced)
- Train on claims from accident years t to t+2, test on t+3, repeat for multiple t values
- Consistent hyperparameter search across methods (same Optuna budget or same grid)
- Primary metric: AUC-PR. Secondary: F2-score at threshold optimised on validation, calibration error, and AUC-ROC for completeness
- A simple baseline that any practitioner could replicate in an afternoon: logistic regression with class_weight='balanced'

The closest public dataset is the IEEE-CIS fraud detection dataset (Kaggle, 2019), which is e-commerce transaction fraud rather than insurance claims. No publicly available UK motor insurance fraud dataset exists. The FIA dataset used in some Kaggle insurance competitions is US-origin.

This is a genuine gap. If an IFB member insurer with a large claims portfolio were willing to release a properly anonymised, temporally-ordered fraud labels dataset — even a 50,000-row sample — it would be the most useful contribution to this literature in a decade. We would write it up and run the benchmark ourselves.

---

## The short checklist

When a fraud ML paper lands in your inbox or on arXiv:

1. **Metric:** Is AUC-PR reported? If only AUC-ROC, the result is incomplete for imbalanced fraud data.
2. **Baselines:** Does the comparison include XGBoost with `scale_pos_weight`? If not, the paper cannot claim to beat the obvious first choice.
3. **Split:** Is the test set temporally later than the training set? If random split, the performance estimate is optimistic.
4. **Data:** Is the fraud prevalence disclosed? Is the jurisdiction and product type the same as yours? Features that are unavailable or unlawful in UK personal lines do not transfer.
5. **Costs:** Is F2 (or a cost-matrix-weighted metric) used, or only F1? A symmetric metric will not optimise the right trade-off.

The Boabang three-stage focal loss curriculum fails items 1, 2, 3, and 4. That does not make the technique worthless — the curriculum training idea has genuine merit for neural network fraud models, and we said so in March. It means the paper cannot be used to decide whether focal loss is worth adopting over your existing XGBoost model. Five questions. If a paper cannot answer them, it is a methods contribution, not an evidence base for operational decisions.

---

**Paper:** Boabang, F. & Gyamerah, S.A. (2025). 'An Enhanced Focal Loss Function to Mitigate Class Imbalance in Auto Insurance Fraud Detection with Explainable AI.' arXiv:2508.02283v2.

**Related:** [Focal Loss for Insurance Fraud Detection — Why You Should Use XGBoost Instead](/2026/03/31/focal-loss-insurance-fraud-detection/) — the full technical comparison and XGBoost implementation guide.
