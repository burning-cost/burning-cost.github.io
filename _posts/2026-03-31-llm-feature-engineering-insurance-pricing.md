---
layout: post
title: "LLM Feature Engineering for Insurance Pricing — What Actually Works"
seo_title: "LLM Feature Engineering for Insurance Pricing: CAAFE, FeatLLM, LLM-FE and the Evidence Gap"
date: 2026-03-31
categories: [machine-learning, techniques]
tags: [LLM, feature-engineering, CAAFE, FeatLLM, LLM-FE, GPT-4, automated-feature-engineering, freMTPL2, Poisson-deviance, interaction-terms, GLM, GBM, EIOPA, fairness, insurance-pricing, python]
description: "Three published frameworks use LLMs to generate tabular features and beat classical search tools on generic benchmarks. None has been tested on an actuarial dataset. We explain why that gap matters, what the techniques do, and where the genuine insurance use case lies."
author: burning-cost
---

There are now three peer-reviewed papers — CAAFE (NeurIPS 2023), FeatLLM (April 2024), LLM-FE (March 2025) — demonstrating that LLMs can generate features for tabular machine learning that outperform classical automated feature engineering tools. The benchmarks are real. The performance gains are modest but consistent. And not one of these papers has tested on an actuarial dataset.

That gap matters more than most practitioners realise. Before explaining what these methods do and where they might actually help, it is worth being precise about what the evidence does and does not show.

---

## What the frameworks do

All three follow the same broad pattern: a language model sees a dataset description and column metadata, proposes new features as Python code, those features are evaluated against held-out performance, and the best are kept. The differences are in how the search is structured.

**CAAFE** (Hollmann, Müller, Hutter — NeurIPS 2023, [arXiv:2305.03403](https://arxiv.org/abs/2305.03403)) uses GPT-4 or GPT-3.5 in an iterative loop. Each round, the model generates a new Python function, the function is applied to training data, the resulting AUC is computed, and the score feeds back into the next prompt as context. Across 14 tabular datasets, GPT-4 improved 11 of them — AUC rising from 0.798 to 0.822 on average. GPT-3.5 only improved 6 of 14. The model dependency is significant. Each generated feature comes with a textual explanation of why the model thought it was useful.

**FeatLLM** (Han, Yoon, Arik, Pfister — [arXiv:2404.09491](https://arxiv.org/abs/2404.09491)) takes a different approach: the LLM generates feature transformation rules once, upfront, using in-context learning. A simple downstream model (linear regression) then uses those rules. Critically, there is no LLM call at inference time — the LLM runs during feature engineering, not scoring. FeatLLM achieves roughly a 10% average error reduction versus TabLLM and STUNT on numerous tabular datasets. The decoupling from inference is commercially important: you do not have a GPT-4 dependency in your live pricing path.

**LLM-FE** (March 2025, [arXiv:2503.14434](https://arxiv.org/abs/2503.14434)) is the most technically ambitious. It runs an evolutionary search: a population of feature engineering programs is maintained across "islands" (m=3), the LLM samples k=2 high-performing programs as in-context examples, and generates variants at temperature 0.8. On the classification benchmark (XGBoost backbone), LLM-FE achieves a mean rank of 1.54 across datasets — compared to CAAFE at 3.82, OpenFE at 3.00, and AutoFeat at 3.09. On regression, mean rank 1.00 versus OpenFE's 2.00. This is the best published result for LLM-based automated feature engineering.

There is a fourth paper worth knowing: a January 2026 preprint ([arXiv:2601.21060](https://arxiv.org/abs/2601.21060)) on human-LLM collaborative feature engineering, which uses a Bayesian neural network surrogate to identify high-uncertainty feature proposals and routes those to human review. Without human input: 7-9% error reduction. With human input: 9-11%. The human-in-the-loop variant is relevant for actuarial practice, where expert sign-off on new rating features is standard.

---

## The operator bias problem

A less-cited paper (arXiv:2410.17787) tested four LLMs — GPT-4o-mini, Gemini-1.5-flash, Llama3.1-8B, Mistral7B-v0.3 — across 27 datasets. The finding is important for anyone thinking about using GPT-4 variants for insurance feature engineering.

GPT-4o-mini and Gemini-1.5-flash show strong bias toward simple operators. Addition dominates. Only five distinct operators account for 90% of all proposed features. Complex aggregation transforms — `GroupByVehicleModel_ThenMean(claims_cost)`, for example — are almost never suggested. This operator bias measurably decreases average predictive performance versus baselines. Smaller open-source models (Llama3.1-8B, Mistral7B-v0.3) show less of this bias.

For insurance pricing, this is not a marginal concern. The features with the highest incremental Gini lift in a mature GBM model are usually exactly the complex aggregation features that GPT-4 will not suggest: vehicle model claim frequency adjusted for driver age profile, postcode-level aggregate loss ratios, interaction terms across three or more rating factors. A tool that reliably proposes `age + mileage` and misses `mean(claims_cost) by vehicle_model × ncd_band` is solving the easy part of the problem.

---

## The actuarial benchmark gap

Here is the honest summary of what has and has not been tested:

**What exists:** LLM-FE tested on a health insurance Kaggle dataset (7 features, 1,338 rows). N-RMSE of 0.381±0.028 for LLM-FE versus 0.383±0.022 for OpenFE. That is a 0.5% improvement — within the error bars, on a dataset with no actuarial structure.

**What does not exist:** No published benchmark on freMTPL2 (the standard European motor benchmark, 679,000 policies). No result showing LLM interaction terms improving Poisson deviance on claims frequency data. No ASTIN, GIRO, or CAS paper with quantitative results on any insurance-specific dataset. No comparison of LLM-generated features against an actuary-designed feature set.

The generic tabular benchmarks use ROC AUC or RMSE on balanced classification or regression problems. Insurance pricing has Poisson/Tweedie targets, exposure-weighted evaluation metrics (Gini coefficient, Poisson deviance, A/E ratios), very high-cardinality categoricals (vehicle models often exceed 10,000 levels in UK motor), and regulatory constraints on which features can be used. None of these characteristics appear in the benchmark datasets used to evaluate CAAFE, FeatLLM, or LLM-FE.

We think there is a non-trivial probability that LLM-generated features do not improve Poisson deviance on a mature motor pricing model. The reason: a well-specified GBM already captures the non-linear relationships that LLMs are most likely to suggest (age curves, vehicle-age decay, mileage bands). The residual improvement from automated feature engineering shrinks as the base model matures.

This experiment is straightforward to run and has not been published. Running CAAFE on freMTPL2 with Poisson deviance as the evaluation metric, comparing generated features against an actuary-designed feature set, is a clear research gap. Until it is done, the generic benchmark results should not be taken as evidence of actuarial uplift.

---

## Where the genuine use case lies

Despite the benchmark gap, there is one actuarial application where LLMs have a credible structural advantage: **interaction term suggestion for GLMs**.

A GLM pricing model can express interaction effects, but identifying which interactions deserve testing is labour-intensive. With $n$ rating factors, the number of candidate two-way interactions is $\binom{n}{2}$, which reaches hundreds for a moderately complex tariff. In practice, actuaries select candidate interactions based on domain knowledge and exploratory analysis. This process is time-consuming, inconsistent across teams, and biased toward interactions that are easy to visualise rather than interactions that are statistically significant.

An LLM can be prompted to propose interaction hypotheses given a structured description of the rating factors and the pricing objective. It will not test those hypotheses — that requires fitting the GLM — but it can prioritise the search. The Actuaries Institute Australia published a practitioner article describing exactly this workflow: LLM generates interaction flag candidates, actuary validates them through standard GLM fitting, the deployed model uses a lookup table, no LLM is in the live pricing path.

The workflow has no peer-reviewed quantitative results. But the mechanism is sound. We have a prior expectation it should work because:

1. The LLM has been trained on insurance pricing literature and will know that `vehicle_age × annual_mileage` is a plausible interaction.
2. The LLM is not constrained by the actuary's visual bandwidth — it can simultaneously consider 20-way taxonomies of rating factors.
3. The validation is actuarially conventional: you fit the interaction term, check the deviance reduction, and keep it if it meets the significance threshold.

This does not require any of the three frameworks above. It is a prompt engineering task.

---

## A worked example

Here is a concrete CAAFE-style workflow adapted for a GLM interaction search. The structure is: describe the dataset and objective, get interaction proposals, validate with a likelihood-ratio test.

```python
import openai
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# --- Step 1: Describe the rating factors to the LLM ---

SYSTEM_PROMPT = """You are an experienced UK motor pricing actuary.
You will be given a list of rating factors for a personal lines motor
frequency model. Propose up to 10 two-way interaction terms worth testing
in a Poisson GLM. For each, give: the interaction, a one-sentence
rationale, and a priority (high/medium/low). Format as JSON."""

rating_factors = """
- driver_age: integer, 17-90
- vehicle_age_years: integer, 0-30
- annual_mileage_band: ordinal 1-7 (1=<5k, 7=>30k)
- ncd_years: integer 0-9, 9=max
- vehicle_group: string, ABI vehicle group 1-50
- region: string, 12 UK regions
- job_title_band: ordinal 1-5 (1=low risk, 5=high risk)
- policy_tenure_years: integer 0-20
"""

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Rating factors:\n{rating_factors}\n\nPropose interactions."}
    ],
    response_format={"type": "json_object"}
)

import json
proposals = json.loads(response.choices[0].message.content)
# Expected: {"interactions": [{"terms": ["driver_age", "vehicle_group"],
#             "rationale": "...", "priority": "high"}, ...]}


# --- Step 2: Validate each high-priority proposal ---

def test_interaction(df: pd.DataFrame, factor_a: str, factor_b: str,
                     target: str = "claim_count",
                     exposure: str = "exposure") -> dict:
    """
    Likelihood-ratio test for a two-way interaction term in a Poisson GLM.
    Returns deviance reduction, p-value, and whether it clears a 0.1%
    significance threshold.
    """
    formula_base = f"{target} ~ {factor_a} + {factor_b} + offset(log({exposure}))"
    formula_inter = f"{target} ~ {factor_a} * {factor_b} + offset(log({exposure}))"

    m0 = smf.glm(formula_base, data=df, family=sm.families.Poisson()).fit()
    m1 = smf.glm(formula_inter, data=df, family=sm.families.Poisson()).fit()

    deviance_reduction = m0.deviance - m1.deviance
    df_diff = m0.df_resid - m1.df_resid
    p_value = 1 - stats.chi2.cdf(deviance_reduction, df_diff)

    return {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "deviance_reduction": deviance_reduction,
        "df_consumed": df_diff,
        "p_value": p_value,
        "significant": p_value < 0.001
    }


# --- Step 3: Rank candidates by deviance reduction per degree of freedom ---

results = []
for item in proposals["interactions"]:
    if item["priority"] == "high":
        a, b = item["terms"][0], item["terms"][1]
        result = test_interaction(claims_df, a, b)  # your DataFrame here
        results.append(result)

results_df = pd.DataFrame(results)
results_df["deviance_per_df"] = (results_df["deviance_reduction"]
                                  / results_df["df_consumed"])
results_df.sort_values("deviance_per_df", ascending=False).head(5)
```

A few notes on this. The LLM is doing semantic prioritisation, not statistical testing. The statistical test is the standard likelihood-ratio test — the same thing an actuary would do manually. The LLM output requires validation before anything enters the model; a plausible-sounding interaction that does not clear the likelihood-ratio threshold is discarded. The workflow would be identical whether the candidates came from an LLM, a domain expert, or a literature review. The LLM speeds up the generation of candidates; it does not change the validation logic.

This is also why the operator bias finding (LLMs prefer addition and subtraction) matters less for this use case than for feature-value generation. Here we are asking the LLM to name which factors to test together, not to construct the interaction mathematically. "Test `driver_age × vehicle_group`" does not require any arithmetic — it requires the LLM to know that younger drivers in higher vehicle groups have disproportionate claim rates. That is well-represented in the training data.

---

## LLMs for governance documentation

There is a second application with a stronger evidence base than feature generation: auto-documenting engineered features.

EIOPA-BoS-25-360 (August 2025) requires that for built or engineered features, "records should exist on how the feature was built and the associated intention." Most pricing teams have dozens of derived features with documentation that ranges from terse code comments to nothing. Bringing a model into compliance with EIOPA's requirement for engineered feature records is tedious work that LLMs are well-suited to.

CardGen (arXiv:2405.06258, May 2024) built a RAG pipeline using Claude 3 Opus and GPT-4-Turbo to auto-generate model cards and data cards. Human evaluation showed LLM-generated cards outperform human-written ones on completeness, objectivity, and understandability — GPT-3.5 scored 5.24/5.23/4.99 against human-written 1.92/2.03/2.49 on those dimensions. CardGen is not insurance-specific, but the task is identical.

A practical implementation for UK pricing teams:

```python
FEATURE_DOC_PROMPT = """You are a UK motor pricing actuary.
Document the following feature for an EIOPA-BoS-25-360 feature record.
Write four sections:
1. What it measures (plain English, one sentence)
2. Why an insurer uses it (actuarial rationale, two sentences)
3. Potential discriminatory risk (one sentence, flag if postcode-correlated)
4. Data lineage (source system and transformation, from the code definition)

Feature name: {name}
Definition: {definition}
Distribution: {stats}
"""

def generate_feature_record(name: str, definition: str,
                             stats: str, client) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": FEATURE_DOC_PROMPT.format(
            name=name, definition=definition, stats=stats
        )}]
    )
    return response.choices[0].message.content
```

The risk is that the LLM generates plausible but inaccurate documentation — particularly on data lineage if the feature definition is ambiguous. Human review before sign-off is essential. This is AI-assisted documentation, not automated documentation. The regulatory value is in the structured output and consistent coverage across all features, not in removing the actuary from the loop.

PRA SS1/23 applies to banks, not insurers — but many large UK insurers have adopted its principles voluntarily as model governance best practice. For teams that follow the SS1/23 framework, that means: LLMs used during model development — including for feature generation or documentation — must appear in the model inventory with their own documentation and vendor dependency chain. If you use GPT-4 to generate interaction candidates, that usage needs to be recorded, even if the LLM never enters the live scoring path.

---

## The discrimination risk you cannot ignore

LLMs trained on internet-scale text encode societal stereotypes. When used to classify vehicle models by "boy racer likelihood" or assign risk scores to occupation categories, those stereotypes enter the pricing model. The insurer is responsible for the outcome; the LLM vendor is not. This is identical to the FCA's existing position on third-party data: "firms need to gain assurance that third-party data used in pricing does not discriminate against customers based on protected characteristics."

Any LLM-generated feature that classifies by subjective characteristic — occupation risk bands, vehicle model personality scores, geographic risk qualifiers — requires the same discrimination-free pricing checks you would apply to a new postcode variable. Marginal correlation analysis against protected characteristic proxies. Shapley attribution to check whether the new feature concentrates value on demographic proxies. A/B comparison of premium distributions across proxied demographic groups.

The features most at risk are exactly the features that LLMs are most naturally inclined to generate: semantic categorisations of nouns (vehicle models, occupations, postcodes) that are correlated with protected characteristics in the training corpus.

---

## What we think is worth trying

**High confidence:** LLM-assisted interaction search for GLM pricing models. The mechanism is sound, the validation is conventional, and the LLM's semantic understanding is a genuine advantage over exhaustive grid search. The appropriate benchmark — does this reduce Poisson deviance more than an actuary's manual candidate list? — has not been published, but the case for trying is strong.

**Medium confidence:** LLM-assisted feature documentation for EIOPA compliance. CardGen's results are encouraging. The task is well-defined. Human review keeps the risk manageable. This is the lowest-barrier entry point for LLM use in pricing model development.

**Low confidence, pending evidence:** Using CAAFE, FeatLLM, or LLM-FE directly on insurance frequency data. The generic benchmark results are real but have not translated to actuarial targets. The operator bias finding is a specific concern for insurance-scale datasets. We would want to see Poisson deviance results on freMTPL2 before recommending any of these frameworks for production use.

**Not recommended without significant additional care:** Semantic scoring of vehicle models, occupations, or geographic categories using an LLM without extensive discrimination testing. The risk is not hypothetical — the FCA's December 2024 Research Note on ML bias and EIOPA-BoS-25-360 both create real audit exposure for firms that deploy LLM-generated proxy features without documented fairness checks.

---

The honest answer to "does LLM feature engineering work for insurance pricing?" is: no one has published the right experiment yet. The available tools have real strengths for semantically structured problems, a documented weakness for the complex aggregation features that matter most for mature pricing models, and a discrimination risk that is higher for insurance than for most other domains. That combination suggests caution about the general-purpose benchmarks and more specific enthusiasm for the interaction-search and governance-documentation applications.

---

*CAAFE: Hollmann, N., Müller, S. & Hutter, F. (2023). 'CAAFE: Context-Aware Automated Feature Engineering.' NeurIPS 2023. [arXiv:2305.03403](https://arxiv.org/abs/2305.03403)*

*FeatLLM: Han, X., Yoon, J., Arik, S. & Pfister, T. (2024). 'FeatLLM: Rethinking Tabular Learning with Foundation Models.' [arXiv:2404.09491](https://arxiv.org/abs/2404.09491)*

*LLM-FE: (2025). 'LLM-FE: Evolutionary Feature Engineering with Language Models.' [arXiv:2503.14434](https://arxiv.org/abs/2503.14434)*

*Operator bias: (2024). 'LLM-based Feature Engineering: Operator Bias and Performance Impact across 27 Datasets.' [arXiv:2410.17787](https://arxiv.org/abs/2410.17787)*

*CardGen: (2024). 'CardGen: Automated Model Card and Data Card Generation via RAG.' [arXiv:2405.06258](https://arxiv.org/abs/2405.06258)*

*EIOPA Supervisory Statement on AI (2025). EIOPA-BoS-25-360. August 2025.*

*PRA Supervisory Statement SS1/23: Model Risk Management Principles for Banks. Bank of England, May 2023 (operative May 2024).*

*Actuaries Institute Australia (2024/2025). 'LLM Feature Engineering in Insurance Pricing.' Practitioner article. [actuaries.asn.au](https://www.actuaries.asn.au/research-analysis/llm-feature-engineering-in-insurance-pricing)*
