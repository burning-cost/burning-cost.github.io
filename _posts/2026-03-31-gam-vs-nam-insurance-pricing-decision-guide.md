---
layout: post
title: "GAMs vs Neural Additive Models: A Decision Guide for UK Pricing Teams"
date: 2026-03-31
categories: [interpretable-ml]
tags: [gam, nam, ebm, explainability, interpretable-ml, neural-additive-models, insurance-pricing, actuarial, explainable-boosting-machine, interpretML, uk-insurance, fca, consumer-duty, insurance-gam, python, pytorch, monotonicity, poisson, tweedie, benchmarking]
description: "arXiv:2510.24601 reviewed 143 papers across 430 datasets and found no consistent accuracy advantage for neural networks over GAMs on tabular data by 2024. What that means for the EBM vs ActuarialNAM vs PIN decision — and when to ignore all three and use a GLM."
---

A systematic review published on arXiv in October 2025 — Doohan, Kook and Burke, [arXiv:2510.24601](https://arxiv.org/abs/2510.24601) — reviewed 143 papers covering 430 datasets across environmental science, engineering and health research from 1998 to 2024. Its conclusion: by 2024, there is no statistically significant accuracy gap between GAMs and neural networks on tabular data. The confidence bands for RMSE ratio, at every network complexity level, include zero.

This matters for insurance pricing teams because the standard justification for black-box gradient-boosted trees over interpretable models has been accuracy. If you cannot demonstrate a meaningful accuracy advantage for XGBoost over an Explainable Boosting Machine, the trade-off looks very different — particularly post-FCA Consumer Duty and with EU AI Act high-risk AI obligations applying from 2 August 2026.

There is an important caveat. We will come to it. But the direction of evidence is clear.

---

## The ICC finding is more important than the headline

The accuracy convergence result is striking. The methodological finding buried in section 4 of the paper is more useful.

Doohan et al. fit a mixed-effects model to their log-ratio accuracy metrics with PaperID as a random effect. The intraclass correlation (ICC) was 0.76 to 0.97 across different metrics. That means 76–97% of the variance in performance ratios across the 143 papers was explained by *which research team ran the comparison*, not by which model family they compared.

The implication is direct: if your pricing team runs an internal benchmark of EBM versus XGBoost, and the two models are tuned by different analysts over different time horizons with different search budgets, the outcome will tell you more about who was more motivated than which model is genuinely better on your data.

Controlled benchmarking is not a nice-to-have. Uncontrolled comparisons are researcher-dominated noise, and this paper demonstrates that rigorously at scale.

---

## What the paper does not tell us

The 430 datasets span environmental modelling, civil engineering and biomedical research. None are insurance. The neural architectures reviewed are shallow — median 21 parameters. The paper does not evaluate EBM, NAM, LocalGLMnet or any of the modern hybrid glass-box approaches that are actually relevant to insurance pricing.

The 'gap has closed' finding may also overstate how competitive standard spline GAMs are against properly tuned deep models on very large datasets. If you have 10 million motor policies and a team with GPU infrastructure, the evidence from this paper does not settle the question of whether a TabNet or FT-Transformer outperforms an EBM.

What it does settle is the abstract argument that you *must* use black-box models to be competitive. That argument has no meta-analytic support on tabular data. The approach to feature engineering and tuning matters more than the model family.

---

## The decision tree

Starting from the library we have actually built and tested — [insurance-gam](https://github.com/burning-cost/insurance-gam), which implements EBM, ActuarialNAM and PIN — here is how we recommend making the choice:

**Does your team have ML infrastructure (PyTorch, GPU, training loop maintenance)?**

No → use **EBM** (InsuranceEBM). Zero-configuration, sklearn API, automatic interaction detection via the FAST algorithm, near-XGBoost accuracy on benchmarks. Default choice for the majority of UK personal lines teams.

Exception: if *n* < 10,000 policies, prefer a traditional **GLM**. At small sample sizes, GAMs and GLMs provide exact confidence intervals via MLE theory and satisfy the actuarial balance property by construction. Neural approaches gain nothing on thin books and lose interpretability.

Yes (ML infra available) → **Are hard monotonicity constraints required?**

Yes → use **ActuarialNAM** (ANAM). The Laub/Pho/Wong architecture (arXiv:2509.08467, NAAJ 2025) enforces monotonicity via Dykstra projection, which is a hard constraint on the shape functions rather than a soft gradient penalty. EBM's `monotone_constraints` parameter applies a gradient regulariser during boosting; the shape function can violate the constraint at individual bins. For age curves and vehicle age curves where a pricing committee will scrutinise any non-monotone region, ANAM is the right tool.

No monotonicity constraints → **Are per-policy exact Shapley values or epistemic uncertainty required?**

Yes → use **PINEnsemble**. The Pairwise Interaction Network architecture is additive over pairs of features, which means Shapley values are exact by construction rather than sampled approximations. The ensemble version (10 models) gives epistemic uncertainty as standard deviation across predictions — useful for underwriting decisions on unusual risks.

No → **EBM** remains the default. PINEnsemble is slower by roughly an order of magnitude and the interpretability gain over EBM is marginal for most applications.

---

## Trade-off matrix

| Property | Traditional GLM | EBM | ActuarialNAM | PIN Ensemble |
|----------|----------------|-----|--------------|--------------|
| Architecture | Canonical link + offset | Cyclic boosted stumps | Per-feature MLP sum | Pairwise interaction network |
| Setup complexity | Low | Very low | High | High |
| Training time | Seconds | Minutes | Minutes–hours | Hours |
| Hard monotonicity | Manual (coefficient sign) | Soft only | Yes (Dykstra) | No |
| Automatic interactions | No | Yes (FAST) | Explicit specification | All pairs |
| Exact confidence intervals | Yes (MLE) | No | No | No |
| Balance property | Yes (canonical link) | No | No | No |
| Exact feature attribution | Yes (shape functions) | Yes (shape functions) | Yes (shape curves) | Yes (exact Shapley) |
| Exposure offset | Native | Via init_score | Native | Native |
| Poisson / Gamma / Tweedie | Yes | Yes | Yes | Yes |
| UK regulatory precedent | Highest | High | Medium (2025 paper) | Low (2024 paper) |
| FCA / EU AI Act compliance | Straightforward | Straightforward | Straightforward | Requires documentation |

---

## When to stop at a GLM

The GLM should be your Phase 1 baseline regardless of which model you subsequently deploy. It gives you the balance property guarantee (sum of predicted frequencies equals sum of observed claims via MLE score equations), exact coefficient confidence intervals, and a reference deviance that everything else must beat.

There are also books where it is the right *final* model:

- **Small n.** Below roughly 10,000 policy-years, the information content is not there to train shape functions reliably. A Poisson GLM with a handful of carefully chosen predictors and offset on exposure will not be meaningfully beaten by EBM on a typical niche commercial or specialty book.
- **Regulatory sign-off required.** UK Lloyd's managing agents and FCA-regulated firms with conservative actuarial functions will find exact CIs on log-linear coefficients more auditable than gradient-based explanations of boosted stumps. The FCA has a long history with GLM; Consumer Duty does not mandate explainability by black-box explanation tools.
- **Balance property is contractual.** Some reinsurance agreements are written against modelled loss estimates that must sum to observed loss. A GLM with canonical link satisfies this without post-hoc adjustment.

---

## The regulatory context

The FCA Consumer Duty (PS22/9, July 2022) requires firms to be able to explain pricing outcomes to customers on request. The EU AI Act (Articles 13 and Annex III, high-risk obligations applying from 2 August 2026) imposes transparency requirements on high-risk AI systems - insurance pricing falls under Annex III. Glass-box models satisfy these requirements directly. XGBoost with SHAP post-hoc explanation satisfies them indirectly, and SHAP approximate values are not identical to the model's actual decision process.

The ICC finding from arXiv:2510.24601 provides indirect support here. If the accuracy advantage of black-box models is not real — if it is primarily an artefact of who tuned the comparison — then there is no meaningful trade-off between accuracy and interpretability. You are not giving up 3% Gini for glass-box compliance. You are giving up researcher-dominated noise.

---

## Controlled benchmarking in practice

Given the ICC finding, every model comparison your team runs should satisfy four conditions:

1. **Same held-out split.** 80/20 random split, fixed seed, applied identically to all models. No model should see the test set during tuning.
2. **Same deviance formula.** Poisson deviance on frequency; Gamma deviance on severity. Not AUC, not RMSE — these are not the actuarial objective function.
3. **Same exposure treatment.** Exposure as offset (log link) for GLM and ANAM; as `init_score = log(exposure)` for EBM; as exposure weight for PIN.
4. **Same hyperparameter search budget.** Time-boxed Optuna or equivalent. If EBM gets 100 trials and XGBoost gets 500, the comparison is not controlled.

[insurance-gam](https://github.com/burning-cost/insurance-gam) currently lacks a benchmark harness that enforces these conditions automatically. That is a gap we intend to close. In the meantime, the four conditions above can be enforced by convention in your modelling scripts.

---

## Practical recommendations

**Default stack for a mid-size UK personal lines team (100K–5M policies):**

1. Fit a Poisson GLM as your baseline. Record train and test deviance. Run the balance check.
2. Fit InsuranceEBM with `interactions='3x'`. Compare test Poisson deviance to the GLM under controlled conditions. If the improvement is less than 0.5% of deviance, question whether the added complexity is justified.
3. If your pricing committee or FCA supervision requires hard monotone age curves: migrate to ActuarialNAM. Budget two to ten times the training time for approximately equivalent or better deviance on large datasets.
4. If GDPR Article 22 right-to-explanation decisions are being made model-by-model: run PINEnsemble in parallel as an audit tool. Do not replace EBM or ANAM; use PIN to verify that the Shapley-derived attributions align with what the primary model is doing.

**Never run an uncontrolled model comparison.** The arXiv:2510.24601 ICC finding is not a theoretical curiosity. If you show your pricing director a slide where EBM beats XGBoost by 2% deviance and it turns out the XGBoost was tuned by a junior analyst for two days while the EBM was tuned by a senior modeller for two weeks, you have learned nothing about the models. You have measured the analysts.

The accuracy/interpretability trade-off argument is substantially dead on tabular data at any scale relevant to UK personal lines. The case for glass-box models is now the affirmative one: direct regulatory compliance, auditable shape functions, hard monotonicity when required, and no dependence on approximate post-hoc explanation tools. We built [insurance-gam](https://github.com/burning-cost/insurance-gam) on this view, and the 2024 evidence base continues to support it.

---

*arXiv:2510.24601: Doohan, Kook, Burke. 'A Systematic Review of GAMs and Neural Networks.' October 2025 (revised January 2026). PRISMA-compliant, 143 papers, 430 datasets, 1998–2024.*

*ANAM paper: Laub, Pho, Wong. 'An Interpretable Deep Learning Model for General Insurance Pricing.' arXiv:2509.08467. NAAJ 2025.*
