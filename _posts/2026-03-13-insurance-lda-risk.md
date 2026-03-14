---
layout: post
title: "Soft Portfolio Segmentation with LDA: Detecting Adverse Selection After a Rate Change"
date: 2026-03-13
categories: [libraries, segmentation, portfolio-management]
tags: [lda, topic-modelling, soft-clustering, portfolio-drift, adverse-selection, python]
description: "Hard clustering assigns every motor policy to exactly one risk group. LDA gives each policy a membership vector across K risk archetypes — and the difference matters enormously when you are trying to detect adverse selection after a rate change."
---

Every pricing team segments their book. The segments go by different names — risk groups, tiers, clusters, bands — but the logic is the same: run k-means or a decision tree, assign each policy to exactly one group, price each group differently. The exercise feels rigorous. It produces a clean table.

The table is wrong, and we can be specific about how.

A 24-year-old male driving a 2019 Vauxhall Corsa in Birmingham is assigned to "Group 3: young urban driver." That label erases the fact that this particular policyholder drives 4,000 miles a year (low), has a Corsa worth £6,500 (low severity), and holds a telematics policy (behavioural signal). His *demographic* profile says high risk. His *behavioural* profile says average. A hard cluster cannot hold both truths simultaneously. It picks one.

Latent Dirichlet Allocation, applied to tabular insurance data, does not pick. It gives the policyholder a membership vector: 60% "urban young driver," 30% "low-mileage occasional user," 10% "fleet-adjacent safe driver." That distribution is his risk fingerprint. The pricing question becomes which archetypes are growing, and at what rate, and why.

---

## The mechanics: policies as documents, modalities as words

The original LDA intuition from Blei, Ng & Jordan (2003) is that a document in a corpus is a mixture of topics. A news article might be 70% politics and 30% economics. The word frequencies in that article are consistent with both topics being active simultaneously.

Jamotton and Hainaut (UCLouvain LIDAM Discussion Paper ISBA 2024/008, published March 2024) made the key observation that this structure is preserved when you move from text to tabular insurance data. The analogy is exact:

| NLP | Insurance |
|-----|-----------|
| Corpus | Portfolio (all policies) |
| Document | Individual policy |
| Word / modality | A specific value of a categorical variable: *female*, *urban*, *age 25–34* |
| Vocabulary | All unique modalities across all categorical variables |
| Topic | Latent risk archetype |
| Document-topic distribution θ_d | Policy's soft membership vector over K risk profiles |

The exchangeability assumption that makes bag-of-words work — the order of words in a document does not matter, only their co-occurrence frequencies — also holds here. Whether *gender=female* appears before *region=urban* or after it is irrelevant. What matters is that these modalities co-occur in a single policy row.

Jamotton and Hainaut test this on the Swedish motorcycle insurance portfolio (62,289 policies, 1994–1998) with K=10 topics. Their LDA model achieves Poisson deviance of 1,276.51 — beating a standard GLM at 1,339.84 with ten clusters against the GLM's 1,269+ cells. LDA is doing more with less.

---

## What `insurance-lda-risk` provides

The library is a Python implementation of the full Jamotton–Hainaut pipeline. There was no public code from the paper; we have released it under MIT at [github.com/burning-cost/insurance-lda-risk](https://github.com/burning-cost/insurance-lda-risk).

```bash
pip install insurance-lda-risk
# or
uv add insurance-lda-risk
```

Five classes, one workflow:

**`InsuranceLDAEncoder`** converts a pandas DataFrame to the sparse count matrix that LDA expects. Continuous variables are discretised via equal-frequency binning (so exposure-dense modalities do not dominate). Missing values become a `__MISSING__` modality rather than being dropped — this is important in real portfolio data where missingness itself carries signal.

**`LDARiskProfiler`** wraps sklearn's `LatentDirichletAllocation` with insurance defaults and returns a (D, K) theta matrix — one row per policy, one column per topic, values are soft membership weights summing to 1.

**`TopicValidator`** evaluates the discovered topics against actual claim outcomes using Poisson deviance — not NLP perplexity, which is meaningless for insurance. This is the actuarial sanity check that the topics are capturing real risk structure, not statistical noise.

**`TopicSelector`** automates the K selection problem using K-fold cross-validation on held-out Poisson deviance and elbow detection. You get a principled answer rather than "we tried K=5 and it looked sensible."

**`PortfolioDrift`** computes Jensen-Shannon divergence between portfolio-level topic distributions across periods. This is the adverse selection detector.

---

## Using it: a UK motor example

```python
import pandas as pd
from insurance_lda_risk import (
    InsuranceLDAEncoder,
    LDARiskProfiler,
    TopicSelector,
    TopicValidator,
    PortfolioDrift,
)

# df has columns: age_band, region, vehicle_group, ncd_years,
# occupation_class, annual_mileage_band, claims, exposure

cat_cols = ["age_band", "region", "vehicle_group", "occupation_class"]
cont_cols = ["ncd_years", "annual_mileage_band"]

# Step 1: encode
encoder = InsuranceLDAEncoder()
X = encoder.fit_transform(df, cat_cols=cat_cols, cont_cols=cont_cols, n_bins=5)

# Step 2: find the right K
selector = TopicSelector(k_range=range(3, 16), n_splits=5)
selector.fit(X, y_claims=df["claims"], exposure=df["exposure"])
selector.plot_elbow()
# returns K=8 for a typical UK motor book of ~50k policies

# Step 3: fit the profiler
profiler = LDARiskProfiler(n_topics=8)
theta = profiler.fit_transform(X)
# theta.shape == (len(df), 8)

# Step 4: validate topics against claims
validator = TopicValidator()
result = validator.validate(theta, y_claims=df["claims"], exposure=df["exposure"])
result.plot_frequencies()
# shows claim frequency per topic — do topics 6 and 7 have
# materially higher frequencies than 1-5? They should if LDA
# found the right structure.

print(result.summary)
#    topic  mean_frequency  share_of_portfolio
# 0      0        0.042            0.31
# 1      1        0.051            0.22
# ...
# 6      6        0.118            0.09
# 7      7        0.143            0.04
```

Topics 6 and 7 here are the high-risk archetypes. They account for 13% of the portfolio by policy count but — if your rates are correctly reflecting this — a disproportionate share of earned premium. If they are *not* earning proportional premium, that is your cross-subsidy.

---

## The use case that justifies implementing this: post-repricing adverse selection detection

Suppose you repriced young urban drivers upwards by 15% in the October renewal cycle. Three months later, you want to know whether it worked — did you deter the worst risks, or did you deter the mediocre ones and retain the genuinely dangerous policyholders who had fewer alternatives?

The traditional answer involves looking at claims development and waiting 12–18 months for any signal. By then the damage is done.

The LDA answer is immediate. Fit the topic model on your pre-repricing book. Now apply the same encoder and profiler to your post-repricing new business and renewals. Use `PortfolioDrift` to compare topic distributions.

```python
drift = PortfolioDrift(profiler)

# pre_df and post_df are the two periods
drift_result = drift.compare(
    pre_df, post_df,
    cat_cols=cat_cols,
    cont_cols=cont_cols,
    exposure_col="exposure",
)

print(f"JSD: {drift_result.jsd:.4f}")
# JSD: 0.073  <-- above 0.05 threshold, worth investigating

drift_result.plot_composition()
# stacked area chart showing topic shares before and after
```

A JSD of 0.073 means the portfolio's risk fingerprint has shifted materially. The `plot_composition()` output will show you which topics gained share and which lost it. If Topic 7 (high-risk urban young driver) *gained* share after your rate increase, that is classical adverse selection: the good risks left, the captive risks stayed.

That finding, with a timeline and a quantified composition shift, is actionable. You can accelerate the repricing, adjust the new business appetite, or revisit the renewal strategy — before claims data confirms the problem at the end of the development year.

---

## M&A due diligence: is the book you're buying the book they say it is?

A second application that Jamotton and Hainaut discuss but do not fully develop: book transfer comparison.

When a UK insurer acquires a motor book — a broker scheme, a white-label arrangement, or a portfolio transfer — they get a data extract and a management account narrative. The narrative says: "predominantly mid-age family drivers, above-average NCD, south-east concentration." The data extract confirms these demographics in the marginal distributions.

What the marginal distributions do not reveal is the *joint* risk structure. A book can have average age band distributions and average region distributions and still have a disproportionate concentration in a high-risk niche that only appears in the joint distribution: young male, high-powered vehicle, urban, low NCD, specific occupation class. Each marginal looks fine. The combination is poison.

LDA sees the joint distribution. Fit a topic model on the acquiree's book, compute the portfolio-level topic distribution, and compare it against your own book using `PortfolioDrift`. A high JSD tells you the books have materially different risk composition even if the marginals appear similar.

```python
acquirer_drift = PortfolioDrift(profiler)  # profiler fitted on your own book

result = acquirer_drift.compare(
    your_df, target_book_df,
    cat_cols=cat_cols,
    cont_cols=cont_cols,
    exposure_col="exposure",
)

# JSD > 0.10: material composition difference
# Ask harder questions before agreeing the purchase price
```

This takes twenty minutes of analysis. The alternative is discovering the composition problem twelve months post-transfer, which is a regulatory and financial problem simultaneously.

---

## FCA Consumer Duty monitoring

There is a less obvious application that UK insurers should pay attention to: Consumer Duty portfolio monitoring.

The FCA's Consumer Duty requirements (PS22/9, in force July 2024 for closed products) include a positive obligation to monitor whether products continue to deliver fair value to the customers they are *actually* reaching, not the customers they were originally designed for.

If your product was designed for the "standard low-risk homeowner" archetype and your book has gradually shifted — through distribution partner changes, aggregator dynamics, or competitor repricing — towards a different risk profile, the product terms may no longer be appropriate for the customers you are actually holding.

`PortfolioDrift` gives you a quantified, explainable signal. "The JSD between our 2023 and 2025 portfolio composition is 0.09. Topic 3 ('high-value property, older owner, low claims frequency') has declined from 28% to 19% portfolio share. Topic 6 ('younger renter, lower sum insured, aggregator sourced') has grown from 12% to 24%." That sentence is auditable evidence that you are monitoring composition, and it tells you where to look at whether your product terms remain appropriate.

---

## Technical caveats worth knowing

The method is not free of assumptions. Two are material in practice.

**The mutual exclusivity problem.** Standard LDA samples each "word" from the full vocabulary. In insurance data, modalities within a single variable are mutually exclusive — a policy cannot simultaneously be *region=urban* and *region=rural*. Jamotton and Hainaut acknowledge this and use standard sklearn LDA anyway because the constrained version is computationally intractable at portfolio scale. The practical consequence is small: the algorithm occasionally assigns small probability mass to impossible modality combinations within a topic, but the topic structures it discovers are empirically valid. The validation step catches truly degenerate topics.

**K selection is noisy.** The elbow in cross-validated Poisson deviance is often shallow. `TopicSelector` returns the K at the detected elbow, but the elbow detection can be unreliable for small portfolios (under 10,000 policies) or portfolios with few categorical variables. The library returns a full elbow plot; inspect it rather than trusting the automated selection blindly. For most UK personal motor books of 50,000+ policies with 5–8 categorical variables, K in the range 6–12 is typical and the elbow is usually clear.

**Exposure weighting is v0.2.** The current version of `LDARiskProfiler` does not exposure-weight the LDA fitting. Policies with higher exposure contribute equally to the topic estimates as policies with one week of cover. This matters if your book has material mid-term adjustment or pro-rata exposure variation. We will add `exposure_weighted=True` in v0.2. For annual policies with minimal MTA, v0.1 is fine.

---

## Why soft membership matters

The hard-clustering alternative — k-means or decision tree segmentation — produces segments that are interpretable but brittle. A policy on the boundary between Segment 2 and Segment 3 gets assigned to one of them arbitrarily; small changes in covariates flip it between segments; the pricing discontinuity at the boundary creates known adverse selection pressure.

Soft membership does not have a boundary problem. The 24-year-old Birmingham Corsa driver with telematics is 60%/30%/10% across three archetypes. If you acquire new data that changes his estimated membership — a mileage update, a mid-term change in vehicle — his risk fingerprint updates smoothly. No cliff edges.

For portfolio monitoring specifically, soft membership is the right tool. Tracking the shift in a portfolio-level topic distribution is a single, stable metric. Tracking the shift in hard cluster assignments requires you to decide whether a policy that moved from Cluster 2 to Cluster 3 is a genuine change or boundary noise. It is usually boundary noise.

---

## The library

[`insurance-lda-risk`](https://github.com/burning-cost/insurance-lda-risk) implements the full Jamotton–Hainaut (2024) pipeline in Python with an sklearn-compatible API. 128 tests passing. Dependencies are standard: scikit-learn, scipy, numpy, pandas, matplotlib.

The paper that motivates it is freely available at the UCLouvain DIAL repository (ISBA 2024/008). Read the paper, then use the library. The paper explains the statistical foundations; the library removes the implementation work.

```bash
pip install insurance-lda-risk
```

Source: [github.com/burning-cost/insurance-lda-risk](https://github.com/burning-cost/insurance-lda-risk)

---

**Related articles from Burning Cost:**
- [Synthetic Insurance Portfolios with Actuarial Fidelity](/2026/03/09/insurance-synthetic/)
- [Full Predictive Distributions for Insurance Pricing](/2026/03/05/insurance-distributional/)
- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/)
