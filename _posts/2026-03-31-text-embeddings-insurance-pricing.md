---
layout: post
title: "Text Embeddings for Insurance Pricing — When Do They Actually Help?"
date: 2026-03-31
categories: [machine-learning]
tags: [NLP, text-embeddings, BGE-M3, sentence-transformers, PCA, LightGBM, commercial-lines, high-cardinality, fairness, BERT, insurance-pricing, feature-engineering, Dong-Quan-2025, Troxler-Schelldorfer-2024]
description: "Text embeddings can compress 13,000 business categories into 24 useful features and add 2-3% Gini lift on commercial lines. For UK personal motor FNOL they do essentially nothing. A guide to where the technique genuinely helps, how to build the pipeline, and what will get you in trouble."
author: burning-cost
---

Most tutorials on text embeddings for insurance pricing treat the technique as universally applicable. They are wrong. The answer is: it works well on commercial lines business descriptions, plausibly on rich claims text, and essentially not at all on UK personal motor FNOL. The distinction matters before you spend three sprints building a pipeline.

The clearest current evidence comes from Dong & Quan (arXiv:2507.21112, 2025) — commercial lines, Carpe Data alternative data, three applications tested on real production data. One of those applications is worth building today. The other two are narrower in scope. Meanwhile, Troxler & Schelldorfer (BAJ 2024) show strong results on 400-word bilingual accident descriptions that have almost no analogue in UK personal lines.

We are going to work through what the literature actually says, build the embedding → PCA → tabular pipeline, and be clear about where you will be wasting your time.

---

## The problem that makes embeddings worth considering

Insurance pricing has a categorical variable problem. Commercial lines underwriting uses business descriptions — SIC codes, NAICS codes, free-form business type fields — that can run to thousands of distinct values. A UK SME insurer might see 13,000 distinct entries in a business category field across five years of policy data. Most cells have fewer than 30 observations. One-hot encoding this into a GLM produces a matrix nobody can validate. K-means clustering produces clusters that look coherent to a human but that do not differentiate loss experience.

This is the specific problem Dong & Quan address, and the results are worth examining directly.

Their dataset: 13,287 distinct business categories from Carpe Data's commercial lines portfolio. They embedded each category description using BGE-M3 (1,024 dimensions), reduced to 24 components via PCA, and fed those 24 features into a LightGBM model alongside the structured rating factors. Gini coefficient: 0.38 versus 0.37 for the baseline — a 2.7% relative improvement. RMSE improved by −0.69%. Neither number is enormous. Together they are material for a commercial book where the previous approach was treating 13,000 categories as a flat lookup.

For comparison, they also tried Word2Vec + k-means with 50 clusters. It did not improve Gini at all. The clusters produced by k-means on Word2Vec embeddings are thematically coherent — related industries end up together — but thematic coherence and risk differentiation are not the same thing. A cluster containing "artisan bakery", "food wholesaler", and "catering supplier" makes intuitive sense and is useless for pricing.

BGE-M3 works because it encodes semantic content in a way that preserves fine-grained distinctions that matter for risk: "roofing contractor" and "interior decorator" end up far apart in embedding space even though both are construction trades.

The paper's other two applications are narrower but worth knowing about. First, de-biasing of star ratings: TextBlob sentiment extracted from review text replaces the ordinal star rating as a feature, on the grounds that raw star ratings carry systematic biases (anchoring, recency, cultural response-scale differences) that the sentiment signal is less susceptible to. Second, unsupervised NAICS classification — LDA topic modelling plus RAKE keyword extraction to assign unclassified businesses to NAICS codes. Both are data quality interventions rather than pure pricing uplift plays.

---

## Where embeddings help: a realistic assessment

Before building anything, it is worth mapping where the technique has genuine actuarial support versus where it is theoretical.

**Commercial lines business descriptions: 1–3% Gini uplift.** This is the Dong & Quan finding. Text encodes risk-relevant semantic content not captured by structured codes. Worth doing if you have high-cardinality business category fields with thin per-cell data.

**Claims text — adjuster notes, 400-word accident descriptions: 2–5% Gini.** Troxler & Schelldorfer (BAJ 2024) tested BERT [CLS] embeddings on bilingual accident descriptions averaging 400 words. The pipeline works. Fine-tuning improves over frozen pretrained weights. The catch for UK practitioners: UK FNOL descriptions are typically 3–20 words, not 400. "Rear-ended at junction" does not give a BERT model much to work with.

**Workers' compensation and specialty lines claims: 2–5%.** Xu, Manathunga & Hong (Variance 2023) used BERT on truck warranty claim descriptions for frequency and severity modelling — "significant improvement in accuracy and stability", in their words. The Manathunga & Doan (Variance 2026) follow-on found GPT-4 and Gemini matching specialised BERT zero-shot on workers' comp dispute prediction when given sufficient context.

**Claims triage and fraud flagging: up to 10–15% AUC.** The best-evidenced use case — and the one that is claims management rather than prospective pricing. Adjuster notes accumulated in days 1–10 post-FNOL are the most predictive text field for final cost outcome. Dimri et al. (AmFam, 2019, 2022) built production systems on this. The signal is real and large.

**UK personal motor FNOL for pricing: near zero.** If you are a personal lines motor actuary wondering whether to add FNOL text features to your frequency model, the answer is no. The description is too short, the information is largely already captured in structured fields, and the regulatory overhead is significant for near-zero expected gain.

---

## The pipeline

For the commercial lines case that works, here is the pipeline. The same architecture applies to longer claims text with minor modifications.

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# -----------------------------------------------------------------------
# Step 1: embed business descriptions
# BGE-M3 for production commercial work; all-MiniLM-L6-v2 for prototyping.
# Do NOT use OpenAI API embeddings — UK GDPR Article 28 data processor
# requirements make this difficult for policyholder data without explicit
# data processing agreements that most insurers do not have in place.
# -----------------------------------------------------------------------

model = SentenceTransformer("BAAI/bge-m3")

# business_descriptions: list of strings, one per policy
# Typical: "commercial cleaning contractor", "artisan bakery - retail only"
embeddings = model.encode(
    business_descriptions,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # BGE-M3 recommendation
)
# shape: (n_policies, 1024)


# -----------------------------------------------------------------------
# Step 2: PCA dimensionality reduction
# Fit ONLY on training data. Transform train and test separately.
# Fitting on the full dataset before the train/test split is the
# most common mistake in published tutorials on this pipeline.
# -----------------------------------------------------------------------

scaler = StandardScaler()
pca = PCA(n_components=24, random_state=42)

X_train_emb = scaler.fit_transform(embeddings[train_idx])
X_train_pca = pca.fit_transform(X_train_emb)

X_test_emb = scaler.transform(embeddings[test_idx])
X_test_pca = pca.transform(X_test_emb)

print(f"Explained variance (24 components): "
      f"{pca.explained_variance_ratio_.sum():.1%}")


# -----------------------------------------------------------------------
# Step 3: concatenate with tabular features
# -----------------------------------------------------------------------

X_train_tabular_pca = np.hstack([X_train_tabular, X_train_pca])
X_test_tabular_pca  = np.hstack([X_test_tabular,  X_test_pca])


# -----------------------------------------------------------------------
# Step 4: fit LightGBM
# -----------------------------------------------------------------------

import lightgbm as lgb

params = {
    "objective": "poisson",
    "metric": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

dtrain = lgb.Dataset(X_train_tabular_pca, label=y_train, weight=exposure_train)
dtest  = lgb.Dataset(X_test_tabular_pca,  label=y_test,  weight=exposure_test,
                      reference=dtrain)

model_lgb = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dtest],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)
```

The 24 PCA components become anonymous continuous features — `text_pc1` through `text_pc24`. They are opaque by design; what matters is whether they carry signal.

How many components? Dong & Quan used 24. The right answer for your data is the scree plot: plot `pca.explained_variance_ratio_` and find the elbow. Typically 15–30 components captures most of the useful variance for insurance text. More than 50 rarely helps and can hurt regularisation.

---

## Model selection

**BGE-M3** (BAAI/bge-m3, 1,024 dimensions, ~570M parameters). The model Dong & Quan used, and the right choice for commercial lines business descriptions. It handles multilingual text well, supports longer sequences than BERT, and its semantic similarity calibration is strong. Runs on CPU for batch work but slow — plan for overnight preprocessing on large portfolios.

**all-MiniLM-L6-v2** (384 dimensions, 22M parameters). For rapid prototyping, for short claims text, for anything where you need to iterate quickly. Five times faster than BGE-M3 on CPU. Performance is meaningfully worse on complex business descriptions but fine for shorter text.

**llmware/industry-bert-insurance-v0.1**. An insurance-domain BERT. Exists. We have not validated it on actuarial tasks and would not use it until someone publishes a proper evaluation. Domain-specific pre-training helps when domain vocabulary diverges materially from general English — insurance jargon does, but not so dramatically that an unvalidated model is safer than a well-validated general one.

**OpenAI text-embedding-3-large or similar API embeddings.** Do not use these for policyholder data without legal sign-off. UK GDPR Article 28 requires a data processor agreement with any third party you send personal data to. Most insurers do not have one in place with OpenAI, and the standard API terms are not designed for regulated financial services data. Run embeddings on-premise.

---

## Five production gotchas

**1. PCA data leakage.** Fit the PCA on training data only. Transform test data using the already-fitted PCA. This sounds obvious; it is routinely wrong in tutorials. If you fit PCA on the full dataset before splitting, the test-set PCA components are informed by test observations — your validation metrics are flattering lies.

```python
# Right
pca.fit(X_train_emb)
X_train_pca = pca.transform(X_train_emb)
X_test_pca  = pca.transform(X_test_emb)

# Wrong — do not do this
X_all_pca = pca.fit_transform(np.vstack([X_train_emb, X_test_emb]))
X_train_pca = X_all_pca[:len(train_idx)]
X_test_pca  = X_all_pca[len(train_idx):]
```

**2. UMAP non-determinism.** UMAP is sometimes suggested as an alternative to PCA for dimensionality reduction. It is not deterministic across runs unless you pin every relevant seed. PCA is deterministic given the same input data and implementation. For a production pricing model, determinism is not optional. Stick with PCA.

**3. Short text noise.** Two-word descriptions ("plumber", "retailer") produce unreliable embeddings. A single word does not give the model enough context to place it accurately in embedding space. For policies where the business description is very short, consider using the SIC code description as a supplement, or apply a minimum description length filter. In Dong & Quan's dataset the descriptions were rich; your CRM data may not be.

**4. PCA versioning.** When you retrain the model, you will likely refit the PCA on newer data. PC1 from the new PCA is not the same feature as PC1 from the old — the components rotate with each fit. You cannot compare coefficients between model versions, cannot reuse old monitoring thresholds, and must treat a PCA refit as a material model change for governance purposes. Serialise and version your PCA objects alongside the model artefact.

```python
import pickle, hashlib

with open("pca_v2_3.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("pca_v2_3.pkl", "rb") as f:
    pca_hash = hashlib.sha256(f.read()).hexdigest()

print(f"PCA artefact SHA256: {pca_hash}")
# Record this hash in your model card
```

**5. Fairness audit.** This is the one that genuinely matters. Text embeddings encode everything in the training corpus, including language patterns that correlate with protected characteristics. A business description that says "family-run halal restaurant" encodes religion and potentially ethnicity. An embedding that compresses 13,000 business categories into 24 components may be compressing demographic proxies alongside the risk-relevant signal.

The regulatory exposure is real: FCA DP23/3 states firms should not use AI that "embeds or amplifies bias"; the Equality Act 2010 applies to indirect discrimination from AI features; Consumer Duty requires outcomes monitoring.

Before deploying, test each PCA component for proxy discrimination:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def detect_proxies(pca_features: np.ndarray, proxy_labels: np.ndarray,
                   threshold_auc: float = 0.65) -> dict:
    """
    For each PCA component, test whether it predicts a protected characteristic
    proxy (e.g., postcode deprivation quintile, IMD decile, estimated ethnicity
    from ONS area-level data).

    Flag any component with AUC > threshold_auc for investigation.
    """
    results = {}
    n_components = pca_features.shape[1]

    for i in range(n_components):
        component = pca_features[:, i].reshape(-1, 1)
        clf = LogisticRegression(max_iter=200)
        clf.fit(component, proxy_labels)
        auc = roc_auc_score(proxy_labels, clf.predict_proba(component)[:, 1])
        results[f"pc{i+1}"] = {
            "auc": round(auc, 4),
            "flagged": auc > threshold_auc,
        }

    flagged = [k for k, v in results.items() if v["flagged"]]
    if flagged:
        print(f"WARNING: components {flagged} predict proxy with AUC > {threshold_auc}")
        print("Consider: adversarial debiasing, dropping flagged components, or "
              "documenting justification for retention.")
    return results
```

No published paper in the actuarial literature has audited sentence-transformer embeddings for proxy discrimination in a UK insurance context. If you are the first team to do this properly and find something interesting, publish it.

---

## The Avanzi alternative

Before building the embedding pipeline for high-cardinality categoricals, it is worth knowing there is a competing approach.

Avanzi et al. (ASTIN Bulletin, 2024) proposed GLMMNet — a hybrid that trains a neural network to estimate random effects for high-cardinality categorical variables, using the GLM likelihood rather than a neural loss. The random effect for each business category is learned jointly with the GLM parameters. No separate text preprocessing, no PCA, no fairness audit on opaque feature vectors.

GLMMNet requires the high-cardinality variable to be structured as a grouping factor rather than a text description. For cases where you have a clean structured code (SIC, NAICS) attached to each policy, it deserves direct comparison with the embedding approach. The embedding approach wins when the free-text description carries information beyond what the structured code captures — which it usually does for commercial lines, where insureds describe their businesses in ways that no code taxonomy fully captures.

---

## What it does not do

**It does not fix thin data.** If you have 13,287 business categories and the rarest 8,000 each appear fewer than 10 times in your training data, an embedding model compresses them into the representation space, but you are still estimating risk from a small number of observations. Embeddings improve how you represent the categorical variable; they do not create observations you do not have.

**It does not explain itself.** The 24 PCA components have no actuarial interpretation. PC1 is "the direction in embedding space that captures the most variance in business description semantics." That is not a rating factor you can describe to a policyholder or a regulator. If explainability is your primary constraint, use the LLM structured extraction approach instead — a local Llama model that reads the text and outputs structured fields like `has_public_liability_exposure: true` or `outdoor_work_fraction: 0.7`. Those fields are auditable; the PCA components are not.

**It does not work on any text.** The signal comes from text that encodes risk-relevant content not already in your structured data. Commercial lines business descriptions meet that bar. UK personal motor FNOL text, at 3–20 words, largely does not.

---

## Summary

For commercial lines pricing with high-cardinality business type fields, sentence-transformer embeddings → PCA → LightGBM is worth building. The Dong & Quan evidence is solid, the pipeline is not complicated, and a 2–3% Gini improvement on a commercial book is meaningful.

For UK personal lines motor, the expected gain is near zero and the fairness audit overhead is real. Do not bother.

For claims triage — post-FNOL classification and reserve signalling, not prospective pricing — embeddings from adjuster notes are the most productive application in the literature. The Dimri et al. work on days 1–10 adjuster notes is the strongest actuarial evidence for any text embedding application we have reviewed.

The thing that will cause you problems is not the model. It is the fairness audit you skip because you are trying to ship before quarter end.

---

## References

- Dong, P. & Quan, Z. (2025). InsurTech Innovation Using Natural Language Processing. arXiv:2507.21112.
- Troxler, A. & Schelldorfer, J. (2024). Actuarial applications of natural language processing using transformers. British Actuarial Journal. arXiv:2206.02014.
- Xu, S., Manathunga, V. & Hong, D. (2023). Framework of BERT-based NLP models for frequency and severity in insurance claims. Variance, 16(2).
- Manathunga, V. & Doan, D. (2026). Predicting workers' compensation dispute outcomes with large language models. Variance, 19.
- Dimri, A. et al. (2022). A multi-input multi-label claims channeling system using insurance-based language models. Expert Systems with Applications.
- Avanzi, B. et al. (2024). GLMMNet for high-cardinality categorical variables in insurance pricing. ASTIN Bulletin.
- FCA & PRA (2023). Discussion Paper DP23/3: Artificial intelligence and machine learning.
