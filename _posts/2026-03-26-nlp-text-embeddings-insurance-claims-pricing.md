---
layout: post
title: "Text Embeddings on Claims Data: The Pipeline, the Papers, and the Limits"
date: 2026-03-26
categories: [techniques]
tags: [NLP, embeddings, sentence-transformers, claims, BERT, transformers, GLM, GBM, pricing, reserving, fairness, FCA, proxy-discrimination, python]
description: "How to turn insurance claims descriptions into GLM features using sentence-transformer embeddings and PCA. What Troxler & Schelldorfer (2024, BAJ) actually showed, what the Kaggle evidence teaches us, and where the technique genuinely earns its complexity cost — and where it does not."
---

The signal is real. FNOL descriptions, adjuster narratives, and claims handler notes contain information that tabular fields do not: peril type, injury severity, liability context, fraud indicators. A claim entered as "third party injury, rear-end collision, claimant represented" and a claim entered as "minor shunt, no injuries reported, claimant settled same day" will have the same structured fields — vehicle type, coverage, location — but very different loss distributions. The text knows the difference. The standard pricing model throws it away.

The question is not whether that information exists. It is whether the extraction pipeline is worth the complexity, regulatory scrutiny, and data quality requirements it brings. The honest answer is: sometimes yes, sometimes emphatically no, and which one depends on what line of business you are working with and what the text actually says.

---

## The pipeline

The standard workflow has four steps:

**1. Collect and clean the text.** Claims systems store free-text in multiple fields: FNOL description, adjuster notes, settlement narrative. You want the field with the most substantive content captured closest to event occurrence. Adjuster notes are usually richest but may be written weeks later. FNOL descriptions are written at first notification and are often brief — sometimes very brief.

**2. Embed with a sentence transformer.** Pass each claim's text through a pre-trained sentence transformer model to produce a fixed-length dense vector. The `sentence-transformers` library (Reimers & Gurevych, 2019) wraps the Hugging Face ecosystem and provides good out-of-the-box models. For English-language UK claims text, `all-MiniLM-L6-v2` (80MB, 384 dimensions) is a reasonable starting point — small enough to run on CPU, sufficient vocabulary coverage. For multilingual books, `paraphrase-multilingual-mpnet-base-v2` handles 50+ languages from a single model.

**3. Reduce dimensionality.** Raw embeddings are 384-768 dimensions. Most of that is noise for any specific insurance application. PCA to 8-20 components is the standard reduction. The Kaggle community implementation of the actuarial loss estimation problem used `paraphrase-distilroberta` (768d) → PCA(20), producing 20 tabular features that fed a LightGBM model. Twenty components is a reasonable upper bound; in practice you often find that 8-12 components capture the meaningful variance.

**4. Use the reduced vectors as features.** The PCA components are continuous predictors. In a GLM they enter as main effects; in a GBM they enter the same way as any continuous feature. The interpretation is opaque — component 1 might capture severity/triviality of the described injury; component 2 might partially separate liability-contested from admitted claims — but you cannot read that off the coefficients. You need additional work (topic modelling or inspection of which claims score high/low on each component) to understand what you are using.

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def claims_text_features(
    texts: list[str],
    n_components: int = 12,
    fitted_pca: PCA | None = None,
) -> tuple[np.ndarray, PCA]:
    """
    Embed claims text and return PCA-reduced features.
    Pass fitted_pca to transform holdout/production data consistently.
    """
    embeddings = encoder.encode(
        texts,
        batch_size=256,
        show_progress_bar=False,
        normalize_embeddings=True,
    )  # shape: (n_samples, 384)

    if fitted_pca is None:
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embeddings)
    else:
        reduced = fitted_pca.transform(embeddings)
        pca = fitted_pca

    return reduced, pca


# Training
train_texts = claims_df_train["fnol_description"].fillna("").tolist()
text_feats_train, fitted_pca = claims_text_features(train_texts, n_components=12)

# Holdout / scoring
test_texts = claims_df_test["fnol_description"].fillna("").tolist()
text_feats_test, _ = claims_text_features(test_texts, fitted_pca=fitted_pca)
```

One thing to get right from the start: fit the PCA on training data only, then transform everything else with the fitted object. This is obvious in hindsight but easy to do wrong when you are iterating quickly on a notebook.

---

## What the academic research actually shows

The most directly relevant paper for UK actuaries is Troxler and Schelldorfer (2023, published British Actuarial Journal 2024): "Actuarial Applications of Natural Language Processing Using Transformers." At 47 pages with 33 figures, it is a detailed tutorial with two worked case studies, not just a methods comparison. Andreas Troxler is at AT Analytics; Jürg Schelldorfer is at Swiss Re.

Their two datasets: car accident descriptions averaging around 400 words (bilingual English/German, Swiss Re internal data), and shorter property insurance claim descriptions. On the car accident data, they demonstrate that BERT transfer learning outperforms bag-of-words approaches, and that domain-specific fine-tuning improves on generic BERT. For multilingual books, they show a single multilingual BERT model handles English and German without separate training runs — useful if you write UK and continental European risks on the same system.

Their embedding pipeline uses the [CLS] token pooled output from BERT (768 dimensions) as the claim-level representation, then feeds this into downstream regression or classification. The regression targets are actuarial: claim cost or loss ratio, not general NLP benchmarks. That alignment with actuarial outcomes is what makes the paper worth reading rather than a generic ML tutorial.

The limitation to note: the car accident descriptions at ~400 words per claim are unusually rich. Swiss Re commercial lines claims narratives are not UK motor FNOL. That caveat is not in the abstract; it matters for how you interpret the results.

Xu, Manathunga and Hong (2023, *Variance*, Vol. 16, Issue 2) is the other key reference: "Framework of BERT-Based NLP Models for Frequency and Severity in Insurance Claims." They embed claim descriptions into a neural network regression for claim severity and show the BERT-enriched model beats the tabular-only baseline when text is available. Again: when text is available — the conditional qualifier that carries most of the practical weight.

---

## The Kaggle evidence

The Kaggle "Actuarial Loss Estimation" competition (closed, ~90,000 synthetic workers compensation claims with claim description text) provides the clearest public benchmark of the full pipeline. Worth examining because competition settings force honest head-to-head comparisons.

The winning solution (Yi Li, PRISM, 91.3% accuracy, best of 3,633 submissions) used rule-based NLP — specifically negation detection to correctly identify body parts involved in injuries, rather than transformer embeddings. "Not" and "no longer" in front of a body part flip the diagnosis. That turned out to be more predictive than the geometry of the embedding space.

The winning approach being rule-based is the most important lesson from this competition. It reflects something real: for well-structured workers compensation claims text with consistent terminology, you can often extract the signal you need with a targeted rule rather than an opaque 768-dimensional projection. Rules are auditable; embeddings are not. When you can write the rule, write the rule.

The community implementation that does use transformer embeddings is also instructive: `paraphrase-distilroberta` → PCA(20) → LightGBM additional features. This is the closest thing to a public reference implementation of the pipeline. The PCA(20) rather than PCA(12) choice reflects the richer vocabulary in workers compensation descriptions; there is more linguistic variance to capture.

---

## Topic modelling versus embeddings

Before reaching for transformers, consider topic modelling — LDA or NMF on bag-of-words representations. It is faster, more interpretable, and sometimes sufficient.

Topic modelling works well when claims fall into a small number of distinct narrative types: fire, flood, liability, theft. Each topic becomes a probability or a hard assignment, and the topic ID is a categorical feature in your GLM. The topic assignments are human-interpretable — you can look at the top words per topic and name them.

Embeddings win when the variation within a claim type matters. "Slipped on wet floor, minor bruising" and "slipped on wet floor, fractured hip" will have similar topic assignments but very different embedding geometry once the model has learned what "fractured hip" means in a severity context. For severity models where the description details predict the quantum of loss, embeddings capture information that topic models miss.

Our view: start with topic modelling. If it saturates — if the inter-topic variation explains the explainable variance and within-topic residuals do not correlate with text content — then move to embeddings. Most UK commercial pricing teams will not need to go beyond topic modelling for their first iteration.

---

## Where it actually works

**Large commercial lines with detailed loss descriptions.** Marine, aviation, energy, large property — lines where the adjuster writes a substantive narrative at each development stage. These descriptions contain information about liability complexity, subrogation prospects, engineering surveys. A 300-word description of a machinery breakdown claim contains more extractable signal than any tabular field on the risk. The Troxler/Schelldorfer setting is close to this.

**Workers compensation with structured injury descriptions.** The Kaggle data demonstrates this. When there is a consistent vocabulary around body parts, incident types, and injury severity, NLP features add lift. The vocabulary is rich but domain-specific, which means general-purpose embeddings may need fine-tuning.

**Liability claims for individual reserving.** If you are building the kind of individual RBNS reserving model described in Wüthrich (2025, arXiv:2603.11660), text from adjuster notes at each development period can supplement the quantitative features — claims incurred and cumulative paid — that drive the linear recursion. The textual signal of "counsel instructed, quantum disputed" versus "matter agreed, awaiting court date" carries information about ultimate cost that is not yet in the numbers.

---

## Where it does not work

**Motor FNOL with three words.** A meaningful fraction of UK motor FNOL descriptions are: "Third party collision", "Rear end shunt", "Parked car damaged." These are below the minimum information content for embeddings to add value. The embedding will represent them faithfully, but faithfully representing three words gives you very little. You will pick up noise from minor phrasing differences, not signal about loss severity. We have seen teams spend weeks on a pipeline to learn that "rear end shunt" and "rear-end shunt" embed slightly differently. That is not actionable information.

**Books where text is not collected at claim opening.** Some systems record the initial FNOL in a structured drop-down (claim type, cause code) and only start free-text recording after the claim is allocated to a handler. In that case, you have text but it is written days or weeks after the event, potentially by a handler who has already seen the quantum. The text becomes post-hoc commentary on the numbers rather than an independent signal, and using it in a pricing context raises circularity questions.

**Small portfolios.** The signal-to-noise ratio in embedding features is low enough that you need meaningful volume to detect it reliably. On a 2,000-claim dataset, the 12 PCA components will fit whatever the training data looks like and may not generalise. The Xu et al. (2023) and Troxler/Schelldorfer (2024) results are from datasets with tens to hundreds of thousands of claims. Do not extrapolate to your niche product.

---

## The FCA angle: proxy discrimination

This is the issue that should be on every UK pricing actuary's checklist before deploying text embeddings, and it is largely absent from the academic literature.

FNOL descriptions are written by claims handlers, and the vocabulary of claims descriptions is not neutral. Research in NLP fairness has consistently shown that text generated by or about individuals correlates with demographic proxies — sometimes directly (postcode mentioned, property type, family structure implied by description), sometimes through language patterns that differ across socioeconomic groups. Embedding a claims description produces a vector that may carry information about the claimant's protected characteristics even when no such information was intentionally included.

Under FCA Consumer Duty, firms must consider whether their pricing models produce outcomes that are inconsistent with fair treatment. Using embedding features derived from free text without testing them for proxy discrimination is a compliance gap. The test is not complicated in principle: stratify your portfolio by protected characteristic proxies (where you have them) and check whether the embedding features have differential predictive power across strata in a way that cannot be explained by the legitimate risk signal.

If your embedding features are predictive because they capture injury severity — that is legitimate. If they are predictive partly because adjuster language differs systematically for certain claimant types — that is not, and it may not even show up in a standard lift chart.

The [insurance-fairness](/2026/03/22/fca-proxy-discrimination-python-testing-guide/) library provides the tooling for this analysis. Run it on your text features before you run it on production data. The documentation has an example for continuous features that applies directly to PCA-reduced embeddings.

---

## Which embedding model to use

For UK English claims text, the practical choices in rough order of preference:

`all-MiniLM-L6-v2`: 80MB, 384 dimensions, runs on CPU in reasonable time for batch processing. Good general English coverage. Our default recommendation for a first pass.

`all-mpnet-base-v2`: 420MB, 768 dimensions, stronger semantic accuracy than MiniLM but 5x slower. Use if MiniLM is not extracting the variance you need.

`paraphrase-multilingual-mpnet-base-v2`: For books with non-English text (UK branches of European insurers, Lloyd's international programmes). 50+ languages, 768 dimensions.

Fine-tuned domain models: Troxler/Schelldorfer show that fine-tuning on insurance-domain text improves performance. This requires labelled insurance claims data at scale and infrastructure to train/serve the model. Most UK pricing teams are not there yet, and the gain from fine-tuning over a pre-trained model is incremental for claim-level severity prediction when the descriptions are in conventional English.

We would not recommend `text-embedding-ada-002` or other OpenAI API embeddings for pricing use: you are sending claims data to a third-party API, which has data governance implications under UK GDPR, and the embeddings are not locally reproducible when the model is updated.

---

## A note on short inputs

The sentence-transformer models were trained on sentence-length inputs. When your claims descriptions are 3-8 words, the model has very little to work with and the embeddings will cluster tightly with high cosine similarity to each other — not because the claims are similar risks, but because short generic phrases occupy a small neighbourhood in embedding space.

One practical mitigation: concatenate multiple text fields before embedding. If you have FNOL description, claim type code (as text), and a brief injury description field, concatenating them gives the model more to work with: "Rear end shunt | Motor TP injury | Whiplash claim soft tissue" is more informative than "Rear end shunt" alone. The concatenation is crude but it works.

Another option for very short text: use topic modelling rather than embeddings, and treat the topic distribution as a prior that the embeddings subsequently refine. We have not seen this combination tested formally in the actuarial literature, but it is methodologically sound.

---

## The summary position

Text embeddings on claims descriptions are a real technique with real evidence behind them. They are not a plug-in component that improves every pricing model. The conditions for success are: text-rich descriptions (200+ words, or shorter but specific and domain-consistent vocabulary), large enough claim portfolios that embedding features can be selected against a genuine out-of-sample test, and willingness to do the proxy discrimination analysis that the FCA's framework requires.

The Troxler/Schelldorfer (2024) paper is the right starting point for any actuary wanting to understand what transformers can do for claims analysis. Read it with the caveat that their datasets are richer than most UK motor or household claims systems produce. The Kaggle actuarial loss estimation community notebooks provide a working implementation of the paraphrase-distilroberta → PCA(20) pipeline that you can adapt.

Start with topic modelling. If the topics saturate, move to embeddings. Run the fairness checks before you run the model in production. And if your FNOL descriptions average fewer than ten words, the pipeline is not ready for your data — the data is not ready for the pipeline.

---

*The sentence-transformers pipeline described here is available in the [insurance-llm-features](https://github.com/burning-cost/insurance-llm-features) repository (forthcoming). The fairness testing workflow for continuous embedding features uses [insurance-fairness](https://github.com/burning-cost/insurance-fairness).*
