---
layout: post
title: "Synthetic Insurance Data That Preserves Correlations: MICE-RF"
date: 2026-03-25
categories: [techniques, synthetic-data]
tags: [synthetic-data, mice, random-forest, privacy, gdpr, eu-ai-act, glm, freMTPL2, imputation, python]
description: "Havrylenko et al. (2025) show that MICE with random forests outperforms CTGAN and VAEs on the freMTPL2 benchmark. We explain why it works, where it fails, and how to run it."
---

Regulatory demand for synthetic insurance data has moved from theoretical to operational. GDPR Article 89 provides the lawful basis for pseudonymisation as a safeguard when personal data is processed for statistical purposes; EU AI Act Article 10 (in force since August 2024) adds the requirement that training data for high-risk AI systems be "relevant, representative, free of errors and complete." Both together create a practical problem: you cannot hand a consultancy your raw policyholder file to build or validate a pricing model, but the synthetic alternatives that existed three years ago destroyed the correlation structure that makes the data useful.

Havrylenko, Käärik, and Tuttar (arXiv:2509.02171, September 2025) ran a systematic comparison of synthetic generation methods on the French freMTPL2 motor dataset (678,013 policies, 9 rating factors, claim count as response). Their result is useful and a little surprising: MICE with random forests beats CTGAN and VAEs on GLM coefficient fidelity, by a wide margin, and is substantially simpler to implement. Their best MICE variant produces a pairwise column MAE of 0.00061 against the real data; CTGAN produces 0.00292 — nearly five times the error.

This post explains why MICE-RF works, where it does not, and how to run it in Python.

---

## Why naive approaches fail

The failure mode for naive synthetic data methods is not marginal distribution error — it is correlation destruction. If you sample each column independently from its fitted marginal, you get plausible-looking driver ages, plausible vehicle groups, and plausible bonus-malus scores. What you do not get is the correlation structure: the fact that young drivers with high-group vehicles and low NCD years cluster together more than their marginal distributions would predict.

This matters because the rating factors in a GLM are not independent. A model trained on independently sampled synthetic data estimates coefficients in a near-diagonal covariance regime; the real data has substantial off-diagonal terms. The estimated relativities will be wrong — not randomly wrong, but systematically wrong in ways that depend on which factors are correlated in the real book.

Simple parametric synthesis (fit a multivariate Gaussian, sample from it) does marginally better but imposes a strong assumption: Gaussian copula dependence. Insurance data is not Gaussian-copula-dependent. Bonus-malus scores and claim counts have highly non-linear tail dependence that a Gaussian copula cannot capture.

Neural methods — CTGAN, TVAE, and their descendants — learn the joint distribution without parametric assumptions. The problem, as Havrylenko et al. demonstrate, is that learning quality degrades on tabular data with mixed types (categorical + continuous + count + exposure) at typical insurance dataset scales. CTGAN's mode-specific normalisation, designed for multimodal continuous distributions, does not handle Poisson claim counts well. On the freMTPL2 benchmark, CTGAN's M₁ score (sum of absolute differences from true GLM coefficients) is 383.63. MICE_PART_SYN achieves 33.94 — an order of magnitude better.

---

## How MICE-RF works

MICE (Multiple Imputation by Chained Equations) was designed for missing data, not synthetic data generation. The key insight in Havrylenko et al. is that synthesis is structurally the same problem: you want to draw plausible values for each variable conditional on all the others.

The algorithm for the MICE_FULL_SYN variant (the fully synthetic version):

1. **Amputation.** Mark 75% of cells in the dataset as missing at random. You now have a partially observed matrix.
2. **First-pass imputation.** For each column, fit a random forest on the observed rows (both observed and imputed values from previous iterations) and impute the missing cells from the RF's predictive distribution — not the point prediction, but a draw from the distribution. This is what makes it probabilistic rather than deterministic.
3. **Second amputation.** Mark the remaining 25% (the cells that were real in step 1) as missing.
4. **Second-pass imputation.** Impute those cells the same way.

The result is a fully synthetic dataset. Every cell was at some point imputed by a random forest conditioned on the remaining columns. The RF captures non-linear relationships and interactions between variables without requiring you to specify them. It handles mixed types naturally: categorical, continuous, count, and binary columns all enter as predictors.

The chained equations structure is the key architectural choice. Rather than fitting a single joint model (which is computationally intractable for high-dimensional mixed-type data), MICE cycles through each variable in turn, treating it as the target and all others as predictors. With enough Gibbs sampling iterations, the chained conditionals converge to the correct joint distribution — in theory. In practice, four to five iterations is typically sufficient.

The choice of RF as the imputation engine (rather than the original MICE default of predictive mean matching or Bayesian linear regression) is what handles the non-linearities. Bonus-malus score and vehicle group have a non-linear interaction in the real data; a linear imputation model would miss this and produce a synthetic dataset with only the linear component of that relationship.

---

## Insurance-specific considerations

The freMTPL2 dataset has some properties that stress any synthetic generation method, and MICE-RF handles them imperfectly.

**Zero-inflation.** Claim counts are zero-inflated — over 93% of policies have zero claims in any given year. The RF imputation model for claim count will see this and produce imputed values concentrated near zero, which is correct in aggregate. But the conditional distribution of non-zero claims is harder to capture. A policy with a bonus-malus score above 100, vehicle group 15, and a young driver should have a higher probability of a non-zero imputed count than the marginal zero-inflation rate. Whether the RF captures this depends heavily on tree depth and the number of real non-zero observations in the training partition. On freMTPL2, with only 5.37% claim frequency, the RF has limited signal for the positive tail.

**Exposure weighting.** The freMTPL2 exposure variable (years of cover) should affect claim count through a rate relationship: claim count ~ Poisson(λ × exposure). MICE-RF does not enforce this relationship explicitly. It learns whatever association exists in the data between the exposure column and the claim count column. If the dataset is large enough and the relationship is strong enough, the imputed values will approximately respect it. But approximate is not the same as actuarially correct. Our [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) library enforces the exposure relationship explicitly at generation time; MICE-RF does not.

**Categorical cardinality.** The freMTPL2 vehicle brand column has multiple levels. RF handles this natively as a factor. Higher-cardinality categoricals (occupational codes in employers' liability, vehicle make/model at the seven-digit level) will degrade RF performance as the number of split candidates increases. In practice we would encode high-cardinality categoricals with target encoding or frequency encoding before fitting the imputation model, then decode after.

**Temporal structure.** freMTPL2 is a cross-section. Longitudinal insurance data (multi-year policies, repeat customers, development triangles) has temporal dependency that MICE-RF treats as just another covariate correlation. This may be acceptable for some use cases and wrong for others.

---

## The code

This uses scikit-learn's `IterativeImputer` (which implements MICE) with `RandomForestRegressor` as the imputation estimator. `IterativeImputer` is still behind the experimental flag in scikit-learn as of 1.5.

One practical limitation: `IterativeImputer` takes a single estimator and applies it to all columns. This means all columns — including categoricals — are imputed with a regressor rather than a classifier. For integer-encoded categoricals this is usually acceptable (you round the output), but it is not quite right statistically. A proper mixed-type implementation would fit a classifier for categorical columns and a regressor for continuous ones; `IterativeImputer` does not support this natively. For a production implementation on a dataset where categoricals dominate, consider `miceforest` (a Python port of the R `missForest` package) which handles this correctly. The code below works well for freMTPL2-style data where categoricals are a minority.

```python
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder


def mice_rf_synthesise(
    df: pd.DataFrame,
    amputation_frac: float = 0.75,
    categorical_cols: list[str] | None = None,
    n_estimators: int = 100,
    max_iter: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    MICE-RF synthesis following the MICE_FULL_SYN approach of
    Havrylenko, Käärik & Tuttar (arXiv:2509.02171, 2025).

    Amputes amputation_frac of cells, imputes with RF, then amputes
    the remaining (1 - amputation_frac) fraction and imputes again.
    Returns a fully synthetic DataFrame with the same schema as df.

    df must be numeric before calling this — encode categoricals first.
    Pass categorical_cols to have those columns rounded after imputation.
    """
    rng = np.random.default_rng(random_state)
    categorical_cols = categorical_cols or []

    df_arr = df.to_numpy(dtype=float)

    # --- Stage 1: amputate amputation_frac of all cells ---
    mask_stage1 = rng.random(df_arr.shape) < amputation_frac
    df_missing = df_arr.copy()
    df_missing[mask_stage1] = np.nan

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        ),
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=True,  # draw from RF leaf distribution, not point prediction
    )

    imputed_stage1 = imputer.fit_transform(df_missing)

    # --- Stage 2: amputate the cells that were real in stage 1 ---
    df_missing2 = imputed_stage1.copy()
    df_missing2[~mask_stage1] = np.nan  # reveal only the previously-imputed cells

    imputed_stage2 = imputer.fit_transform(df_missing2)

    # Round integer/categorical columns (encoded as floats, need integer output)
    cat_indices = [df.columns.get_loc(c) for c in categorical_cols]
    for col_idx in cat_indices:
        imputed_stage2[:, col_idx] = np.round(imputed_stage2[:, col_idx])

    return pd.DataFrame(imputed_stage2, columns=df.columns)


# --- Example usage on freMTPL2-style data ---

# Assume df is the real motor dataset with columns:
# exposure, area, vehicle_power, vehicle_age, driver_age,
# bonus_malus, vehicle_brand, vehicle_gas, density, claim_count

categorical = ["area", "vehicle_power", "vehicle_brand", "vehicle_gas"]

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_encoded = df.copy()
df_encoded[categorical] = enc.fit_transform(df[categorical]).astype(float)

synthetic_encoded = mice_rf_synthesise(
    df_encoded,
    amputation_frac=0.75,
    categorical_cols=categorical,
    n_estimators=100,
    max_iter=5,
    random_state=42,
)

# Decode categoricals back; clip to valid range before inverse transform
synthetic_encoded[categorical] = enc.inverse_transform(
    np.clip(
        synthetic_encoded[categorical].to_numpy(),
        0,
        None,
    )
)

synthetic_df = synthetic_encoded
```

A few things to note in this implementation.

`sample_posterior=True` in `IterativeImputer` is what makes this properly probabilistic. Without it, the imputer returns the RF's mean prediction for each missing cell. With it, it draws from the RF's leaf-node distribution — the set of training observations that fall into the same leaf as the query point. This introduces the stochasticity needed to generate genuinely different synthetic samples rather than a smoothed version of the original data. This parameter was added in scikit-learn 1.0.

The two-stage amputation is the MICE_FULL_SYN design from the paper. MICE_PART_SYN (their best-performing variant by M₁ score) amputates 75% and then appends the imputed synthetic rows to the original data, producing a mixed real/synthetic dataset. That is useful for augmentation tasks; for pure synthesis, the two-stage approach is what you want.

For a dataset of 678,000 rows with 11 columns and `n_estimators=100`, expect roughly 3 hours on a standard workstation. This is consistent with the paper's timing observations. For development work, use a 10% sample to tune settings, then run the full synthesis overnight.

---

## Limitations

**Tail dependence is not well preserved.** MICE-RF captures the body of the joint distribution well. It does not capture extreme-value dependence: the probability that a very high bonus-malus score co-occurs with a very high claim count. The paper reports this as future work ("more complex dependence structures"). For most model development purposes (training and validating a GLM or GBM), this is acceptable. For capital modelling, where you need the joint tail, it is not.

**Privacy guarantees are empirical, not formal.** The paper does not address privacy at all. MICE-RF does not copy real records — the imputation draws from learned conditional distributions, so a synthetic record is a plausible interpolation of the training data rather than a copy of any specific policy. But there is no differential privacy bound, no epsilon guarantee, and no Membership Inference Attack evaluation. "Distance to Closest Record" tests suggest real-vs-synthetic distinguishability is low on freMTPL2, but this has not been demonstrated with adversarial methods. Before relying on MICE-RF output for data sharing with external parties, test it with `anonymeter` or equivalent.

**Computationally heavy on wide data.** freMTPL2 has 11 columns. UK commercial insurance data can have 200+. The chained equations structure means fitting one RF per column per iteration: at 200 columns and 5 iterations that is 1,000 RF fits. With 100 estimators each, this becomes impractical without heavy parallelisation or dimensionality reduction first. The standard approach for wide data is to group related columns and run MICE within blocks, but this requires domain knowledge to design the block structure.

**Data augmentation does not reliably help.** This is the finding from the paper we find most practically important: mixing real and synthetic data for GLM training did not improve model performance in the freMTPL2 experiments. The synthetic data adds noise to the correlation structure, and for a well-specified GLM on a large real dataset, this hurts rather than helps. Augmentation may help in the thin-data case (where the real dataset has fewer than ~10,000 exposures), but this was not tested.

---

## Where this fits against the alternatives

MICE-RF is not the only option. Our own [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) library uses vine copulas with AIC-selected marginals, which preserves tail dependence better and enforces actuarial structure (exposure/frequency relationship, hard constraints on driver age etc.) that MICE-RF ignores. The tradeoff is that vine copulas require more domain-specific setup and do not naturally handle the amputation-imputation use case.

The practical choice depends on what you are trying to do:

- **Sharing a realistic training set with an external party for GLM development:** MICE-RF is a good default. Simple, well-validated on freMTPL2, good coefficient fidelity. Run an anonymeter check before sharing.
- **Generating portfolios for fairness testing or stress testing where correlations must be realistic and constraints enforced:** vine copulas.
- **Capital model inputs where joint tail behaviour matters:** neither. Neither MICE-RF nor vine copulas preserve extreme-value dependence adequately. Use a proper extreme-value model or an empirical bootstrap of the real data with appropriate anonymisation.

The paper is Havrylenko, Käärik & Tuttar, "Synthetic data for ratemaking: imputation-based methods vs adversarial networks and autoencoders," arXiv:2509.02171 (September 2025). The key benchmark is freMTPL2 with 678,013 policies. MICE_PART_SYN achieves an M₁ score of 33.94 versus CTGAN's 383.63 — a real result, not a marginal improvement. We think MICE-RF belongs in every pricing team's synthetic data toolkit, with eyes open about what it does not do.

---

- [Why Generic Synthetic Data Fails Actuarial Fidelity Tests](/2026/03/09/insurance-synthetic/)
- [Does Proxy Discrimination Testing Actually Work?](/2026/03/28/does-proxy-discrimination-testing-actually-work/)
- [Does Conformal Prediction Actually Work for Insurance Claims?](/2026/03/26/does-conformal-prediction-actually-work-for-insurance-claims/)
