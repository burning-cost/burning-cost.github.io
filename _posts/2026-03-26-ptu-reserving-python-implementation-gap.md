---
layout: post
title: "The PtU Reserving Algorithm in Python: Filling the Gap Left by Richman-Wüthrich"
date: 2026-03-26
categories: [reserving, techniques, python]
tags: [reserving, individual-claims, PtU, chain-ladder, RBNS, IBNR, python, sklearn, backward-recursion, motor-bi, employers-liability, richman, wuthrich, insurance-severity, actuarial, uk]
description: "Richman-Wüthrich's one-shot PtU reserving paper (arXiv:2603.11660) ships with R code only. We map the algorithm to Python, explain the censored-claims exposure mechanism that makes it work, and give an honest read on where it earns its place in a UK reserving toolkit."
---

Yesterday we covered [what the Richman-Wüthrich PtU paper empirically shows](/2026/03/25/one-shot-individual-claims-reserving-neural-networks-vs-chain-ladder/) — linear regression beating neural networks on small datasets, claims incurred outperforming cumulative paid as a predictor, and a 44% reduction in RMSEP versus Mack chain-ladder on accident insurance. The response from readers was largely: "The method looks useful. The code is R. Can we have Python?"

No Python implementation of arXiv:2603.11660 exists yet. Not on PyPI, not on GitHub, not anywhere. The paper shipped with 30 lines of base R. We are building the Python port into [`insurance-severity`](/insurance-severity/). This post covers the part that is easy to misread in the paper — the censored-claims exposure mechanism and the backward recursion — and gives the Python implementation sketch.

---

## What "pay-to-use" actually means

The paper's term "pay-to-use" refers to how the algorithm handles the central challenge in RBNS reserving: at the time of evaluation, most claims have only partial development histories. You cannot use a claim in your learning set to fit a development model unless its ultimate is already known. But recently reported claims — the ones where your reserve estimate matters most — have no known ultimate at all.

The PtU exposure mechanism solves this by restricting each development-step model to a cohort where the ultimate is (approximately) observable. At development step *j*, you only train on:

1. Claims with reporting delay *T ≤ j-1* — so the cumulative paid at period *j-1* is observable
2. Claims from accident periods *i ≤ I-j* — old enough that the ultimate is effectively settled

Condition (2) is the "pay-to-use" constraint in practice. You pay for information about recent claims by restricting the learning set to old claims. You use the model trained on old claims to estimate reserves for recent claims. The name is slightly obscure but the mechanism is clean.

This is not a new idea — it is precisely the cohort restriction that makes standard chain-ladder factors internally consistent. What the Richman-Wüthrich framework does is apply the same restriction at claim level and then extend it to arbitrary regression functions rather than a simple ratio of averages.

---

## The backward recursion: why it runs in reverse

The algorithm runs from the longest development period back to the shortest. This is the part that catches most people.

At step *j = J-1* (one period from ultimate), the learning set is fully known: you have claims with their actual ultimates *C_{i,J}*, and you fit a regression of ultimate on current cumulative paid. Use this to predict ultimates for all claims currently at development lag *J-1*.

At step *j = J-2*, you do not have ground-truth ultimates — they are still in the future. Instead, you use the *J-1* predictions from the previous step as targets. This is the key recursive dependency: each step's predictions feed the next step backwards.

```
j = J-1: fit OLS(C_hat_J ~ C_{j=J-1}), using ground-truth ultimates
j = J-2: fit OLS(C_hat_J ~ C_{j=J-2}), using J-1 predictions as targets
j = J-3: fit OLS(C_hat_J ~ C_{j=J-3}), using J-2 predictions as targets
...
j = 0:   fit OLS(C_hat_J ~ C_{j=0}),   using j=1 predictions as targets
```

The prediction at step *j* for a claim in the most recent diagonal is the model for that step applied to current data. The backward structure avoids the compounding errors of forward simulation: you are not predicting future payments period by period, you are predicting the ultimate directly at each step from current data.

---

## Python sketch

The paper's R code is a single script with no encapsulation. The Python structure we are building into `insurance-severity` has four components: data validation, factor computation, the backward regressor, and bootstrap uncertainty.

**Data schema**

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Minimum viable schema — one row per (claim, development_period)
REQUIRED_COLS = [
    "claim_id",           # str/int
    "accident_period",    # int, 1..I
    "reporting_delay",    # int >= 0: periods from accident to first report
    "development_period", # int >= 0: periods since first report
    "cumulative_paid",    # float >= 0
    "claim_status",       # int {1=open, 0=closed}
]

# Optional but strongly recommended for liability lines
OPTIONAL_COLS = [
    "claims_incurred",    # float >= 0: paid + handler's case reserve
]
```

**Cohort filtering — the critical step**

```python
def learning_set(
    df: pd.DataFrame,
    step_j: int,
    evaluation_date_I: int,
) -> pd.DataFrame:
    """
    Return the learning set L_{j-1} for development step j.

    Restricts to claims satisfying:
      reporting_delay <= j-1  (observable at period j-1)
      accident_period <= I-j  (old enough for known ultimate)
    """
    return df[
        (df["development_period"] == step_j - 1)
        & (df["reporting_delay"] <= step_j - 1)
        & (df["accident_period"] <= evaluation_date_I - step_j)
    ].copy()
```

Getting both conditions right is where a naive implementation fails. Missing the `reporting_delay` condition includes claims that were not yet observable at period *j-1*, polluting the learning set with impossible observations. Missing the `accident_period` condition includes recent claims whose ultimates are not yet known, forcing you to use predicted ultimates as targets when you should have ground truth.

**Backward recursion**

```python
from sklearn.linear_model import LinearRegression

def fit_ptu_regressor(
    df: pd.DataFrame,
    J: int,
    I: int,
    features: list[str] = ["cumulative_paid"],
    target_col: str = "ultimate_estimate",
) -> dict[int, LinearRegression]:
    """
    Fit one OLS model per development step, running backwards from J-1 to 0.
    Returns dict mapping step -> fitted regressor.

    At step J-1: target = actual ultimate (C_{i,J})
    At step j < J-1: target = predictions from step j+1
    """
    models = {}

    # Initialise target column with ground-truth ultimates
    # (only available for fully developed claims — accident_period <= I-J)
    df = df.copy()
    df[target_col] = df["actual_ultimate"].where(
        df["accident_period"] <= I - J, other=np.nan
    )

    for j in range(J - 1, -1, -1):
        learn = learning_set(df, step_j=j + 1, evaluation_date_I=I)

        if len(learn) < 10:
            # Too few observations: fall back to scalar CL factor
            models[j] = None
            continue

        X = learn[features].values
        y = learn[target_col].values

        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        models[j] = model

        # Propagate predictions backward: claims at development lag j
        # get their ultimate estimated by this model
        mask = (df["development_period"] == j) & df[target_col].isna()
        if mask.any():
            df.loc[mask, target_col] = model.predict(
                df.loc[mask, features].values
            )

    return models, df
```

**Balance property**

One thing worth preserving: `LinearRegression(fit_intercept=True)` satisfies the balance property automatically. The sum of OLS residuals is zero in-sample, which means the model's aggregate predictions match aggregate observations within the learning cohort. This prevents systematic bias propagating backward through the recursion. If you switch to a neural network or GBM, you need to verify the balance property holds — it will not do so automatically unless you impose it through the loss function or a post-hoc correction.

**IBNR residual**

```python
def ibnr_reserve(
    df_open: pd.DataFrame,
    ptu_ultimates: pd.Series,
    cl_ultimate_by_accident_period: dict,
) -> pd.Series:
    """
    IBNR = CL total - sum of individual RBNS ultimates (eq 3.3).

    ptu_ultimates: per-claim ultimate estimates from the backward regressor
    cl_ultimate_by_accident_period: aggregate chain-ladder ultimate per period
    """
    rbns_sum = (
        ptu_ultimates
        .groupby(df_open["accident_period"])
        .sum()
    )
    cl = pd.Series(cl_ultimate_by_accident_period)
    return (cl - rbns_sum).clip(lower=0)
```

The IBNR component is a residual, not a bottom-up individual estimate. This is a genuine limitation: for long-tailed lines where IBNR is a large fraction of total reserve, this module still depends on chain-ladder for the largest component. What you gain from the individual model is better RBNS accuracy and the ability to explain individual reserve movements — which is often what actuarial sign-off committees actually want.

---

## Feature choice: what the data says

On the accident insurance data (66,639 claims), cumulative paid alone as a feature produces Ind.RMSE = 1.489 for accident year 2, stepping up to 8.218 for accident year 5 as the projection horizon lengthens.

On the liability data (21,991 claims), the feature ranking by Ind.RMSE is unambiguous:

| Feature set | Ind.RMSE |
|---|---|
| Cumulative paid only | 4.265 |
| Paid + claims incurred | 3.154 |
| **Claims incurred only** | **3.089** |

Claims incurred — the handler's running case reserve plus paid to date — reduces individual claim error by 28% versus cumulative paid alone. For a UK employers' liability or motor bodily injury book where handlers maintain case reserves with reasonable discipline, this is the single most actionable finding: add claims incurred to your feature vector and ignore the cumulative payments column if you have to choose between them.

The practical limitation is data availability. Claims incurred requires that your system tracks historical reserve snapshots, not just the current reserve. Many legacy UK claims systems overwrite the case reserve rather than logging each revision. If your system does not have incurred history, you are constrained to cumulative paid and claim status — still better than chain-ladder, but not as strong as the liability headline number suggests.

---

## Why there is no Python implementation yet

The paper was submitted on 12 March 2026 — two weeks old at time of writing. Richman and Wüthrich ship R code because R is the language of choice in continental European actuarial practice and the code is demonstrably short. The companion paper (arXiv:2602.15385, February 2026) also used R.

The [`chainladder-python`](https://github.com/casact/chainladder-python) library handles aggregate triangle methods and has no individual claims module. The `insurance-reserving-neural` library covers neural individual reserving from the Avanzi et al. architecture, which is a different approach entirely — transaction-level simulation rather than one-shot projection.

The PtU individual regression method is unambiguously simpler than Avanzi et al., and the R implementation is around 30 lines. The Python port adds cohort validation, configurable feature sets, and bootstrap uncertainty — call it 600 lines — but the core algorithm is not complex. The main reason nobody has done it yet is that the paper is two weeks old.

We are building this into `insurance-severity` as a `reserving` submodule. The implementation target is:

- `PtUData`: validates and filters the individual claims panel
- `PtURegressor`: backward recursive OLS with configurable features, sklearn-compatible
- `PtUBootstrap`: 500-iteration bootstrap for RMSEP decomposition
- Demo notebook against SPLICE-style synthetic data

---

## UK application: motor BI and EL

The two natural homes for this technique in a UK book are motor bodily injury and employers' liability. Both are long-tailed, both have open/closed status tracked from FNOL, and both typically have handlers maintaining running case reserves with some discipline.

**Motor BI:** UK motor bodily injury claims follow the Civil Liability Act 2018 tariff for soft-tissue injuries (the whiplash reform), but more serious injuries — orthopaedic, neurological, fatal — remain complex long-tail claims with multi-year development. This is exactly the heterogeneous development pattern where individual models win over aggregate chain-ladder. A claim that has been open for three years with substantial payments and an increasing case reserve has a very different development trajectory to a one-year-old claim with minimal payments and a static reserve. Chain-ladder applies the same factor to both. PtU does not.

**EL/PL:** UK employers' liability claims can develop over 20+ years (industrial disease, asbestos). The paper uses a 5-year horizon; UK EL would require 15-20 backward steps. The learning sets at early development steps become very thin — you are only training on claims from accident years more than 15 years ago. We would expect linear regression to remain competitive here precisely because the small learning sets do not support complex models, but the feature stability assumption (similar development patterns across decades) is untested and probably does not hold for the oldest cohorts.

The framework does not provide an individual IBNR module. For UK EL, where IBNR can be 40-60% of total reserve (latent claims, long reporting delays), this is a significant gap. The RBNS component can be estimated individually; the IBNR component falls back to aggregate chain-ladder or frequency-severity methods. That is still progress — better RBNS accuracy reduces the noise in your IBNR residual — but it is not a complete replacement.

---

## What this means for pricing teams

Pricing actuaries tend to think reserving is someone else's problem. We think that is wrong in three ways.

**IBNR loadings in pricing models.** Your pricing model charges for expected claims. For long-tailed lines, a material portion of claims are not yet reported when you price them. The IBNR loading you apply depends on the reliability of your reserving model. If the RBNS component is over-reserved (chain-ladder compounds IBNR contamination into RBNS factors), you are systematically misestimating the proportion of total ultimate that attaches to reported vs unreported claims. PtU's explicit RBNS/IBNR decomposition gives you cleaner inputs for this calculation.

**Individual claim development for pricing by risk characteristics.** If you know that claims for a particular injury type, vehicle class, or demographic cohort develop differently — and you have individual claims data to demonstrate it — the PtU framework lets you condition on those static covariates explicitly. The Richman-Wüthrich extension in Section 5 adds business line, claim type, and reporting delay as covariates. This directly feeds back into pricing segmentation decisions: if claims from certain risk characteristics develop more heavily than the portfolio average, that development difference should appear in your loss cost estimates.

**Claims incurred as a pricing signal.** The finding that handler case reserves are more predictive of ultimate than cumulative payments has a pricing implication. Handler reserve-setting behaviour is a source of noise in your ultimate cost estimates — systematic under-reserving or over-reserving by claim type inflates variance in your training data. Controlling for this in a reserving model is one path; another is using reserve adequacy as a diagnostic on the claims population feeding your pricing GLM.

---

## Limitations: what the paper does not tell you

**Small-data proviso.** The "linear regression beats neural network" finding comes from datasets of 22k-67k claims over a 5-year triangle. It is almost certainly correct for that setting. Whether it holds for a UK motor book with 500,000 claims per accident year and 10 years of development history is genuinely unknown. We would expect the neural advantage to surface at scale, particularly if you add claims text features from adjuster notes — a setting where the PtU feature set can be extended (see our [NLP embeddings post](/2026/03/26/nlp-text-embeddings-insurance-claims-pricing/) for the embedding pipeline).

**Least-squares loss.** OLS minimises squared error. For reserving, under-prediction is asymmetric with over-prediction in regulatory and commercial terms. Solvency II requires a best estimate, not a minimum squared error estimate. In practice these often coincide for central tendencies, but in the tail they diverge. Using a quantile regression variant as the objective — predicting the 50th or 75th percentile rather than the mean — would be straightforward with sklearn and arguably more defensible for reserving purposes.

**Data quality dependency.** The method requires claims incurred history, consistent claim status coding, and reliable reporting delays. UK legacy claims systems vary substantially in what they retain. If your system overwrites case reserves, you lose the incurred feature and fall back to cumulative paid — the weaker predictor. This is a data engineering problem, not a modelling problem, but it is the showstopper in practice for many books.

**Validation burden under Solvency II technical provisions requirements.** A novel reserving methodology requires documented backtesting, comparison against established methods, and sign-off under Solvency II validation requirements — specifically the SS15/16 (Solvency II: technical provisions) standard for the appropriateness and validation of actuarial assumptions used in best estimates. Running PtU individual reserving as a shadow model alongside chain-ladder for two years before seeking regulatory approval for use in technical provisions is the realistic path for firms subject to PRA oversight.

---

## Connection to insurance-severity

The [`insurance-severity`](https://github.com/burning-cost/insurance-severity) DRN module produces per-claim predictive distributions at FNOL — the distributional shape of what a claim is likely to cost at ultimate, given its initial characteristics. The PtU reserving module is a development model — given a claim partway through its development, what will it cost at ultimate.

These are complementary. The DRN gives you the distributional prior at FNOL; PtU updates that estimate dynamically as the claim develops and you observe cumulative payments, case reserve movements, and open/closed transitions. The claims incurred variable that PtU uses as a feature is directly related to the DRN's point estimate updated with development information.

This also means you can use the DRN's uncertainty — its ability to distinguish wide-distribution from narrow-distribution claims — to weight your PtU predictions. Claims where the DRN uncertainty is high at FNOL are precisely the ones where bootstrapped reserve uncertainty should be largest, and a weighted PtU model could reflect this. That is not in the paper; it is the natural extension once both modules are in the same library.

---

## What to do now

If you have UK motor BI or EL individual claims data with a reasonably clean incurred history:

1. Structure your data as a panel (one row per claim per development period) with the schema above. This is the main work — the modelling is straightforward once the data is in shape.
2. Fit the basic linear PtU model using cumulative paid only. Compare total RBNS reserve against your current method. Expect it to be closer to true OLL than aggregate chain-ladder, and expect the error to be lower for recent accident years where the projection horizon is short.
3. Add claims incurred as a second feature. On liability lines this should reduce individual claim RMSE materially.
4. Run the bootstrap — 500 iterations is sufficient — and compare the RMSEP against Mack. If it is lower (it should be), you have the quantitative argument for the sign-off conversation.

The paper is two weeks old. The R code is public. We are shipping Python. If you are implementing this and hit anything broken with the cohort filtering or backward recursion logic, get in touch.

---

**Related posts:**
- [One-Shot Individual Claims Reserving: What the Richman-Wüthrich Paper Actually Shows](/2026/03/25/one-shot-individual-claims-reserving-neural-networks-vs-chain-ladder/) — the empirical results and NN vs linear comparison
- [Conformal Reserve Ranges: Finite-Sample Coverage Guarantees for IBNR Intervals](/2026/03/16/reserve-range-conformal-guarantee/) — distribution-free uncertainty bounds for reserve estimates
- [Text Embeddings on Claims Data: The Pipeline, the Papers, and the Limits](/2026/03/26/nlp-text-embeddings-insurance-claims-pricing/) — adding adjuster narrative as a PtU feature

---

**Paper:** Ronald Richman and Mario V. Wüthrich, *One-Shot Individual Claims Reserving*, arXiv:2603.11660 (March 2026).

**Code:** [`insurance-severity`](https://github.com/burning-cost/insurance-severity) — PtU reserving module in development. Install with `uv add insurance-severity`.
