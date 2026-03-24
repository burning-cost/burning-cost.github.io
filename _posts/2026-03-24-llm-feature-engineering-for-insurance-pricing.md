---
layout: post
title: "LLM Feature Engineering for Insurance Pricing: What the Research Actually Shows"
date: 2026-03-24
author: burning-cost
categories: [techniques]
tags: [LLM, embeddings, feature-engineering, NLP, sentence-transformers, GLM, GBM, insurance-gam, insurance-fairness, text, occupations, claims-notes, python, pricing]
description: "What large language models can genuinely contribute to insurance pricing feature engineering — text embeddings, zero-shot classification, synthetic features — and where the evidence ends and vendor hype begins."
---

Every InsurTech pitch deck from the past two years has contained the phrase "AI-powered pricing." Dig into the technical detail and you will find XGBoost with more features. Usually the same features. Sometimes the same model, renamed.

There is, however, a genuine technique in this space: using large language model embeddings as a preprocessing step to turn unstructured text fields into usable pricing features. The academic foundation is real. The practical tooling exists. The production challenges are also real, and most teams deploying in the UK are not close to solving them.

This post separates what the research actually shows from what vendors claim, provides working Python code, and is honest about where the technique stands today.

---

## What the research says

The foundational work on neural embeddings for insurance pricing is Wüthrich and Merz — specifically their 2023 monograph *Statistical Foundations of Actuarial Learning and its Applications* (Springer), which consolidates a decade of work on neural architectures for frequency and severity modelling. The core insight for our purposes: high-cardinality categorical variables — vehicle make, occupation class, territory — are better represented as learned dense embeddings than as sparse one-hot encodings. This is entity embedding applied to insurance ratemaking, and it works.

The entity embedding idea originates in Guo and Berkhahn (2016, arXiv:1604.06737), who applied it to tabular prediction problems. Shi and Shi (2023, *North American Actuarial Journal*, 27(1), 175–205) formalised it for insurance risk classification, showing significant Gini lift on several portfolio types. Wang, Shi and Cao (2025, NAAJ) extended this to a full nested GLM pipeline with spatial constraints — which is what our [insurance-glm-tools implementation](/2026/03/09/nested-glms-with-neural-network-embeddings-for-insurance/) is based on.

What these papers share: the embeddings they use are *learned from the insurance data itself*, via a neural network trained on claim outcomes. They are not pre-trained general-purpose LLM embeddings. That distinction matters and we will return to it.

The natural extension — using pre-trained transformer embeddings on free-text fields that insurers already collect — is newer and less well documented in peer-reviewed literature. The application to insurance pricing specifically remains sparse as of early 2026: the academic work is concentrated on entity embeddings for structured categoricals (strong evidence base) rather than on sentence-transformer embeddings for unstructured proposal or claims text (weak evidence base, but logically coherent). Both are worth exploring. They are not the same technique.

---

## The three things LLMs can actually do

### 1. Text-to-embedding for unstructured fields

Motor proposals collect vehicle description as a text field. Home insurance applications collect property construction notes. Commercial lines collect risk descriptions. Claims systems collect adjuster notes. None of this gets into a pricing model because it is not structured.

Pre-trained sentence transformer models produce fixed-length vector representations of text. A vehicle description of "2019 Ford Transit Custom 290 L1 H1 Limited" and "Transit Custom van 2019" should produce similar vectors. They do. The similarity is driven by the model's training on hundreds of millions of sentences, which gives it a robust representation of what words mean and how they relate.

The practical workflow: embed the text, reduce dimensionality (PCA or UMAP to 8–16 dimensions), and feed the reduced vectors into your GLM or GBM as continuous features. The model then treats the embedding coordinates as continuous rating factors.

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Load a lightweight model — all-MiniLM-L6-v2 is 80MB, runs on CPU
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def build_text_features(
    texts: list[str],
    n_components: int = 12,
    fitted_pca: PCA | None = None,
) -> tuple[np.ndarray, PCA]:
    """
    Embed a list of text strings and return PCA-reduced features.
    Pass fitted_pca from training to transform a holdout set consistently.
    """
    # Encode in batches to avoid OOM on large datasets
    embeddings = encoder.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
    )  # shape: (n_samples, 384)

    if fitted_pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embeddings)
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        pca = fitted_pca
        reduced = pca.transform(embeddings)

    return reduced, pca


# On a training set
vehicle_desc_train = df_train["vehicle_description"].fillna("unknown").tolist()
text_features_train, fitted_pca = build_text_features(vehicle_desc_train, n_components=12)

# On holdout — use the fitted PCA, do not refit
vehicle_desc_test = df_test["vehicle_description"].fillna("unknown").tolist()
text_features_test, _ = build_text_features(vehicle_desc_test, fitted_pca=fitted_pca)

# Assemble into feature matrix
text_cols = [f"veh_emb_{i}" for i in range(12)]
df_train[text_cols] = text_features_train
df_test[text_cols] = text_features_test
```

The 12 embedding dimensions then enter your GBM or GLM exactly as any other continuous feature. For a GLM, treat them as restricted cubic splines or thin-plate smooths — they are continuous coordinates in a learned semantic space, not inherently linear.

### 2. Zero-shot classification of dirty categoricals

Occupation fields are a persistent problem. The Association of British Insurers maintains a standard occupation list with around 1,500 coded occupations, but collected data is messy: free text entries, abbreviations, misspellings, synonyms. "Plumber" and "Plumbing contractor" are the same risk. "Teacher" and "Secondary school teacher" should map to the same pricing band.

Zero-shot LLM classification solves this without labelled training data. You define the target classes — your occupation pricing bands — and the model assigns each free-text entry to the most appropriate class.

```python
from transformers import pipeline

# Zero-shot classifier using a BART NLI model
# facebook/bart-large-mnli: 1.6GB, production-quality
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1,  # CPU; use device=0 for GPU
)

# Your pricing bands
occupation_bands = [
    "professional/managerial",
    "clerical/administrative",
    "skilled manual trade",
    "unskilled manual",
    "self-employed",
    "student/education",
    "retired",
    "other",
]

def classify_occupation_batch(
    texts: list[str],
    candidate_labels: list[str],
    batch_size: int = 32,
) -> list[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = classifier(batch, candidate_labels, multi_label=False)
        if isinstance(outputs, dict):
            outputs = [outputs]
        results.extend(o["labels"][0] for o in outputs)
    return results


sample_occupations = [
    "Gas safe engineer",
    "Year 3 teacher",
    "MD of small construction firm",
    "HGV driver class 1",
    "Retired headteacher",
]

bands = classify_occupation_batch(sample_occupations, occupation_bands)
for occ, band in zip(sample_occupations, bands):
    print(f"{occ!r:45} -> {band}")

# 'Gas safe engineer'                           -> skilled manual trade
# 'Year 3 teacher'                              -> professional/managerial
# 'MD of small construction firm'               -> self-employed
# 'HGV driver class 1'                          -> skilled manual trade
# 'Retired headteacher'                         -> retired
```

This is not a pricing innovation — it is a data quality fix. You are still using occupation bands as a rating factor. You are just mapping messy input data to those bands more accurately than a lookup table can. The accuracy gain is real: a lookup table fails to classify free-text entries and falls back to a default band; the zero-shot classifier rarely does.

### 3. Synthetic feature generation from claims notes

Claims notes contain structured information that claims systems never extract: liability split narratives, third-party vehicle descriptions, witness counts, weather conditions, road types. An LLM can extract structured features from these notes at scale.

This is the technique with the most genuine upside and the most genuine risk. The upside: if you can reliably extract "single-vehicle accident on motorway, dry conditions, rear impact" from a claims note, you have a richer description of loss causation than the FNOL tick-boxes give you. The risk: LLMs hallucinate. On free-text claims notes, they will confidently extract facts that are not in the note, or misread ambiguous text.

The mitigation is strict structured output with confidence scores, not free-form generation:

```python
from pydantic import BaseModel, Field
from openai import OpenAI  # or Anthropic, or a local Ollama endpoint

client = OpenAI()  # requires OPENAI_API_KEY in environment

class ClaimFeatures(BaseModel):
    n_vehicles_involved: int = Field(ge=1, le=10)
    motorway_incident: bool
    adverse_weather: bool
    third_party_liability_disputed: bool
    extraction_confidence: float = Field(
        ge=0.0, le=1.0,
        description="0=could not extract, 1=clearly stated in note",
    )

def extract_claim_features(note: str) -> ClaimFeatures:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured features from an insurance claims note. "
                    "Only extract what is explicitly stated. If a field cannot be "
                    "determined from the note, set extraction_confidence below 0.5."
                ),
            },
            {"role": "user", "content": note},
        ],
        response_format=ClaimFeatures,
        temperature=0,
    )
    return completion.choices[0].message.parsed
```

The `extraction_confidence` score is essential. Features extracted with confidence below 0.7 should not enter a pricing model: you are adding noise, not signal. Run the extractor across a sample of claims notes where the true structured fields are known — from FNOL data or adjuster records — and validate the extraction accuracy before using the features.

---

## Why most "AI-powered pricing" is not this

Large InsurTech vendors — Duck Creek, Majesco, Akur8, EIS — routinely describe their platforms as AI-powered or LLM-enabled. What this means operationally is almost never LLM feature engineering in the sense described above.

Akur8's core offering is automated feature selection and GLM fitting, with a UI layer over what is essentially regularised regression. This is useful. It is not LLM-based. Their 2024 product updates mention "AI assistance" in the model building workflow, which refers to automated variable screening, not transformer embeddings.

Duck Creek's pricing tools are built around Emblem-equivalent factor tables with workflow tooling. There is nothing LLM-adjacent in the core pricing engine.

The InsurTechs that have genuine LLM integration in pricing pipelines are, as of early 2026, experimental-only. No UK carrier we are aware of is running pre-trained transformer embeddings in production rating. Several are running pilots. The tooling to do this at rating-engine latency — sub-100ms — with the required audit trail does not exist off the shelf.

This is not a criticism of the technique. It is a statement about where the technology sits on the adoption curve.

---

## Limitations you cannot ignore

**Hallucination in extracted features.** LLMs do not know what they do not know. They will extract plausible-sounding facts from ambiguous text. Any feature extracted from claims notes or proposal text by an LLM must be validated against ground truth before use. The extraction pipeline above includes a confidence score for this reason. Use it.

**Regulatory explainability.** The FCA's Consumer Duty requires firms to evidence fair value outcomes. If a pricing model uses 12 PCA components derived from sentence transformer embeddings of vehicle descriptions, you need to be able to explain what those dimensions represent. "Dimension 7 of the PCA of the all-MiniLM-L6-v2 embedding" is not an explanation a pricing committee will accept, and the FCA will not either. The mitigation is to treat embedding-derived features as inputs to a downstream interpretable model — an EBM or [an ANAM via insurance-gam](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — that produces a reviewable shape function for each embedding dimension. This adds engineering complexity but is required for FCA-facing deployment.

**Proxy discrimination risk.** Pre-trained embeddings are trained on general-purpose text corpora that encode societal biases. A sentence transformer trained on web text has absorbed associations between occupation descriptions and gender, between place names and ethnicity. When you embed free-text fields, those associations enter your feature space. The resulting pricing features may proxy for protected characteristics in ways that are not visible in a standard model review. This is a live concern under Consumer Duty and Equality Act Section 19. Any embedding-based feature should go through proxy discrimination testing — the [insurance-fairness](/2026/03/03/your-pricing-model-might-be-discriminating/) framework handles this. Do not skip this step.

**Inference cost at rating scale.** `all-MiniLM-L6-v2` encodes roughly 14,000 sentences per second on an A100 GPU, and around 800 per second on a modern CPU. For a real-time rating engine processing quote requests, encoding each new quote's text fields at query time is too slow without GPU infrastructure. The practical solution for most teams is to pre-compute embeddings at ingestion time and store them in the policy record. This works for proposal text fields. It does not work for live claims notes, which are written during a claim that may span months.

**Embedding drift.** Pre-trained model providers update their models. `all-MiniLM-L6-v2` from 2022 and a hypothetical updated version produce different embedding spaces. If you retrain your pricing model on new embeddings, your historical feature values are no longer comparable without re-embedding. Pin your sentence transformer version in production and treat it as a versioned dependency with the same change management discipline as your pricing model itself.

---

## What to do now

If you are a UK pricing team with free-text fields in your data and no LLM preprocessing, the highest-ROI first step is occupation zero-shot classification. It requires no new infrastructure beyond a one-time batch inference job, it improves existing factor quality without adding new model complexity, and it is straightforward to validate and explain.

The embedding approach for vehicle description or claims notes is higher effort with higher upside, but requires explainability and fairness testing infrastructure to be in place before production use. Most teams are not there yet.

Building an `insurance-llm-features` library for this is on our roadmap. The components — sentence transformer wrapper, PCA pipeline with consistent train/test handling, zero-shot classifier with calibrated confidence, claims note extractor with structured output — are not complicated individually. The value is in the opinionated pipeline that connects them to a downstream GLM or GBM with the appropriate audit outputs. That library does not exist yet. This post is the advance thinking for it.

---

## References

- Wüthrich, M.V. and Merz, M. (2023). *Statistical Foundations of Actuarial Learning and its Applications*. Springer.
- Shi, P. and Shi, K. (2023). Non-life insurance risk classification using categorical embedding. *North American Actuarial Journal*, 27(1), 175–205.
- Wang, Y., Shi, P. and Cao, C. (2025). A nested GLM framework with neural network encoding and spatially constrained clustering in non-life insurance ratemaking. *North American Actuarial Journal*.
- Guo, C. and Berkhahn, F. (2016). Entity embeddings of categorical variables. *arXiv:1604.06737*.
- Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP 2019*. (sentence-transformers library: sbert.net)

---

## Related

- [Nested GLMs with Neural Network Embeddings for Insurance Ratemaking](/2026/03/09/nested-glms-with-neural-network-embeddings-for-insurance/) — entity embeddings trained on insurance data (different approach to the same family of problems)
- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — downstream interpretable models that can consume embedding features with explainable outputs
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — why embedding-derived features need fairness testing before FCA-facing deployment
