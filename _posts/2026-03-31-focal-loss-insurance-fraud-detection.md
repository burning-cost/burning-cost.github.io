---
layout: post
title: "Focal Loss for Insurance Fraud Detection — Why You Should Use XGBoost Instead"
date: 2026-03-31
categories: [fraud, machine-learning, class-imbalance]
tags: [focal-loss, XGBoost, SMOTE, fraud-detection, class-imbalance, FCA, Consumer-Duty, SHAP, cost-sensitive-learning, motor-insurance, Boabang-Gyamerah-2025]
description: "Focal loss is a clever idea from computer vision that does not translate well to tabular insurance fraud data. AUC=0.63 from a three-stage focal loss neural network versus 0.75-0.85 from a well-tuned XGBoost tells you what to use. We explain why, and how."
author: burning-cost
---

A new paper on focal loss for auto insurance fraud detection has just landed on arXiv (Boabang & Gyamerah, 2508.02283, August 2025). The technique is interesting. The results are not good. And the paper accidentally makes the strongest possible case for just using XGBoost.

We think focal loss is the wrong tool for insurance fraud detection on tabular data. Here is why, what works better, and what the regulatory picture means for your choice.

---

## The problem: 1-in-46 claims is confirmed fraud

UK motor insurance fraud is not a rounding error. The ABI's November 2025 report put detected motor fraud at 51,700 claims worth £576m in 2024 — up 5% from 2023. Against 2.4 million total motor claims, that is a detected rate of 2.15%, or roughly 1-in-46.

"Detected" is a floor. Industry consensus is that true fraud runs at 2-3x the detected figure. You are probably training on a 1-in-46 imbalance ratio, but the underlying population is closer to 1-in-20.

Any fraud classifier has to handle this imbalance. The question is how.

---

## What focal loss is, and where it came from

Focal loss was introduced by Lin et al. in 2017 (RetinaNet, ICCV best student paper). The motivation was one-stage object detection in images, where a single forward pass over a 640×640 image might produce 100,000 candidate bounding boxes, of which perhaps 10 contain objects. The model drowns in easy negatives — background patches that are trivially not objects — and the gradient signal from the handful of hard positives gets swamped.

The fix is elegant. Standard cross-entropy: `L = -log(p_t)`. Focal loss adds a modulating factor:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

When the model is already confident about an example (`p_t` close to 1), `(1 - p_t)^gamma` is close to zero and that example contributes almost nothing to the gradient. Hard examples — where `p_t` is low — are up-weighted. `gamma=2` is the default in computer vision applications.

The mechanism is clever. The question is whether it applies to fraud.

---

## The Boabang & Gyamerah paper

The paper extends this with a three-stage curriculum training procedure to improve convergence stability:

**Stage 1** — Convex surrogate: replace `log(p_t)` with `softplus` to ensure a convex loss surface during initialisation.

**Stage 2** — Intermediate: half-weight transition to the non-convex focal loss.

**Stage 3** — Full non-convex focal loss.

The curriculum idea — starting with a convex loss and annealing toward the target loss — is methodologically sensible. Convex optimisation guarantees convergence to a global minimum; focal loss with neural networks on a small dataset can get stuck. The three-stage approach is a reasonable attempt to get the gradient dynamics right.

The results are not convincing:

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Three-stage (theirs) | 0.607 | 0.588 | 0.333 | 0.415 | 0.630 |
| Convex only (alpha=0.1) | 0.573 | 0.669 | 0.086 | 0.145 | 0.532 |
| Non-convex (alpha=0.25) | 0.559 | 0.205 | 0.029 | 0.051 | 0.586 |
| Non-convex (alpha=0.5) | 0.569 | 0.339 | 0.162 | 0.205 | 0.580 |

The three-stage approach wins the internal comparison — the curriculum training does help. But AUC=0.63 is 13 percentage points above a random classifier. That is not a system you would put in production.

The critical gap: the paper does not compare against XGBoost, LightGBM, Random Forest with SMOTE, or any industry-standard baseline. It only compares focal loss variants against each other. That is not enough to conclude focal loss is useful for fraud.

---

## Why tabular fraud data is different from image detection

The analogy to object detection breaks down in three places.

**The imbalance structure is different.** In RetinaNet, the easy negatives are trivially easy — background pixels look nothing like objects. The signal-to-noise problem is spatial. In insurance fraud, the easy negatives are genuine routine claims, but the hard cases are not rare fraud claims that a well-trained model would find if only it could focus on them. The hard cases are ambiguous claims — legitimate claims that pattern-match fraud, and fraudulent claims that have been structured to look legitimate. Downweighting easy examples does not help when the difficult examples are intrinsically ambiguous, not just rare.

**Neural networks on tabular data already struggle.** Tree-based models — XGBoost, LightGBM — consistently outperform neural networks on tabular data. This has been demonstrated repeatedly since Grinsztajn et al. (2022, NeurIPS). The 39,981-row dataset in the Boabang paper is not large enough to overcome this structural disadvantage. A neural network with focal loss is solving a harder optimisation problem with fewer data than XGBoost needs.

**Gamma needs to be much lower.** The default `gamma=2` from computer vision implies aggressive downweighting of easy examples. With a 1:46 imbalance ratio on a 40,000-row dataset, you have roughly 850 fraud cases. Even `gamma=1` leaves the gradient almost entirely driven by the 40 or so fraud cases the model currently gets wrong. The gradient signal from hard fraud examples is very noisy at this sample size. Boabang uses `gamma=1-2` but does not report sensitivity analysis.

---

## What XGBoost with cost-sensitive weighting does instead

XGBoost handles class imbalance directly via `scale_pos_weight`, which scales the gradient contribution from positive (fraud) examples:

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# At 2.15% fraud rate: scale_pos_weight ≈ 46
n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()
scale_pos_weight = n_negative / n_positive  # ~46 for UK motor

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',   # AUC-PR is more informative than AUC-ROC at extreme imbalance
    early_stopping_rounds=50,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
```

This is conceptually equivalent to `alpha` weighting in focal loss — it scales the loss contribution by class frequency. It is not equivalent to `gamma` modulation, which focal loss adds on top. But for tabular fraud data, `alpha` weighting is almost certainly the right lever. The tree-based gradient computation already handles hard examples differently from neural network backpropagation.

The focal loss equivalent for a neural network, for comparison:

```python
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction='none'
        )
        p_t = torch.exp(-bce)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

# Then you need a neural network, a training loop, early stopping,
# learning rate scheduling, batch normalisation, dropout tuning...
# For 40,000 rows of tabular data, this is considerably more work
# than XGBoost for results that are typically worse.
```

This is not a knock on focal loss as a technique. For the problem it was designed for — dense object detection in images — it is the right tool. For a 40,000-row tabular dataset with 2% fraud prevalence, you are adding complexity that does not help.

---

## SMOTE: useful but limited

SMOTE (Synthetic Minority Over-sampling Technique) is the other standard approach, and it is worth being precise about where it helps and where it does not.

SMOTE generates synthetic fraud examples by interpolating between real fraud cases in feature space. The limitation for insurance fraud is that many variables are categorical: claim type, cover type, policy channel, vehicle category. SMOTE interpolation is meaningless on categorical features — you cannot interpolate between "comprehensive" and "third party fire and theft." You need SMOTENC (the extension for mixed data types) or you drop categoricals before applying SMOTE.

The more serious issue is data leakage. Applying SMOTE to the full training set before splitting inflates performance estimates. The synthetic fraud cases end up in both training and test folds. Apply SMOTE only within training folds during cross-validation, never before splitting.

When done correctly — SMOTENC within training folds — SMOTE can add 2-4% AUC lift over unweighted training. It is a legitimate technique. But it is not clearly better than XGBoost with `scale_pos_weight`, which requires no synthetic data at all.

---

## The FCA Consumer Duty problem

There is a regulatory dimension that the Boabang paper does not address, and it matters for UK practitioners.

FCA Consumer Duty (PS22/9, in force July 2023) requires firms to act to deliver good outcomes for retail customers. For fraud detection, this means: when you decline a claim on fraud grounds, you need to be able to explain why. A decision that cannot be explained is a decision you may not be able to defend.

SHAP on XGBoost is well-understood. For a given claim, you can show which features drove the fraud score, in what direction, by how much. "Your claim was flagged because the reported incident time is inconsistent with the vehicle's GPS trace, which contributes 0.18 to the fraud score" is a defensible explanation.

A neural network trained with focal loss does not give you this directly. SHAP works on neural networks too — via DeepSHAP or GradientSHAP — but the attributions are less stable and harder to validate than tree SHAP, which has a mathematically exact fast implementation. The Boabang paper runs SHAP on its model, but reports only qualitative feature rankings — no numerical attribution values — which suggests the explanations are not robust enough to report precisely.

For a UK motor insurer subject to Consumer Duty, the explainability gap is a real constraint, not a theoretical one.

---

## What the industry actually uses

For context: UK insurers do not, in general, run bespoke fraud classification models in the way this research assumes. The IFB's Exploration platform — built on Shift Technology's Force product, with the two organisations announcing a deepened partnership in 2025 — handles cross-insurer fraud network detection. The model is graph-based: it looks for connected clusters of claims sharing claimants, vehicles, third parties, or legal representatives. Individual claim scoring is supplementary, not primary.

Smaller insurers without IFB Exploration access are the firms where a bespoke XGBoost fraud model would be deployed. For those teams, the practical starting point is:

1. XGBoost with `scale_pos_weight` set to the training set imbalance ratio
2. AUC-PR as the primary evaluation metric (more informative than AUC-ROC at 2% prevalence)
3. Threshold calibration via Platt scaling rather than defaulting to 0.5
4. SHAP attributions for Consumer Duty explainability
5. A written validation of the model against the FCA's Consumer Duty fair outcomes requirements before deployment

Focal loss belongs on that list only if XGBoost AUC-PR < 0.65 and you have reason to believe the problem is specifically the hard-example gradient issue rather than data quality, feature engineering, or insufficient training data.

---

## The curriculum training idea is worth salvaging

We do not want to be entirely negative about the Boabang paper. The three-stage convex-to-non-convex curriculum is a legitimate technique for stabilising neural network training on heavily imbalanced tabular data. If you are building a neural network fraud model — perhaps because you need to integrate unstructured data (FNOL text, images) with tabular features in a single model — the curriculum approach is worth trying. Start with convex surrogate loss, anneal to focal loss over the first 20% of training epochs. It does stabilise convergence.

The issue is the framing: this should be "a way to make focal loss training more stable" not "a new state of the art for insurance fraud detection." The comparison against XGBoost baselines is missing, and that missing comparison makes the paper's conclusions impossible to evaluate.

---

## Our recommendation

Use XGBoost with `scale_pos_weight`. It is faster to train, better calibrated on tabular data, and easier to explain. The threshold-calibration step that most teams skip is often worth more than switching to focal loss. Set your decision threshold not at 0.5 but at the point that maximises F1 on your validation set, and re-examine that threshold annually as fraud patterns shift.

Focal loss is a tool from a different domain, applied to a problem it was not designed for, on data that is too small to let a neural network compete with gradient-boosted trees. AUC=0.63 is the result. That number tells you what you need to know.

---

**Paper:** Boabang, F. & Gyamerah, S.A. (2025). 'An Enhanced Focal Loss Function to Mitigate Class Imbalance in Auto Insurance Fraud Detection with Explainable AI.' arXiv:2508.02283v2.

**Data sources:** ABI (November 2025), *Insurance Fraud Statistics 2024*. Banulescu-Radu (2024), ARIA Prize paper, *Journal of Risk and Insurance*.
