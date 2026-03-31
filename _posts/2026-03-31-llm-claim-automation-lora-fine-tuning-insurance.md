---
layout: post
title: "Fine-Tuning LLMs on Claims Data: What a 2M-Record Warranty Study Tells Us"
date: 2026-03-31
categories: [llm]
tags: [LLM, fine-tuning, LoRA, claims, NLP, DeepSeek, transformers, claims-automation, FCA, GDPR, motor, warranty, operations]
description: "Mo et al. (arXiv:2602.16836, Feb 2026) fine-tuned DeepSeek-R1-Distill-Llama-8B on 2 million automotive warranty claims. The LoRA-tuned model hit 81.5% structured-output accuracy versus 62.1% for the best prompt-only approach. We look at what that result means — and doesn't mean — for UK motor and property claims operations."
---

A paper published in February 2026 by Mo and Quan at PCMI Corporation demonstrates something that anyone who has tried to squeeze structured output from a reasoning LLM will recognise immediately: prompt engineering alone is not enough, and the gap between prompting and fine-tuning on domain data is larger than it looks on toy examples.

The paper — "Claim Automation using Large Language Model" (arXiv:2602.16836) — is worth reading carefully because it has real data (2 million automotive warranty claims), a thoughtful evaluation framework, and results that are directly transferable to UK motor claims operations even though the training set is US automotive warranty. We will cover what they did, what the numbers mean, and where we think this applies in a UK personal lines context.

---

## The task: generating structured corrections from claim narratives

The claims processing domain has a well-established three-field schema called 3C: Complaint, Cause, Correction.

- **Complaint**: what the customer reported ("vehicle won't start, electrical fault warning illuminated")
- **Cause**: the technician's diagnostic conclusion ("corroded battery terminal, failed cell")
- **Correction**: the repair action performed ("battery replaced, terminal cleaned and retorqued")

Mo et al. train a model to take Complaint and Cause as input and generate a structured Correction recommendation. The output is not free text — it has to conform to a parseable schema that downstream systems (parts ordering, cost estimation, settlement triggering) can consume without human intervention. Getting format compliance right is therefore as important as getting the semantics right.

This is an operations automation problem, not a pricing problem. The direct users are claims handlers, not pricing actuaries. We will come back to the pricing connection, but we want to be clear about what the paper actually does before making any larger claims about it.

---

## Four model configurations and what they show

The evaluation compares four configurations on 200 human-verified cases:

| Model | Configuration | Format % | Validity % | Accuracy |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | No prompt | 0% | 8% | 56% of valid |
| DeepSeek-R1-Distill-Llama-8B | Prompt engineering | 6.5% | 29.5% | 64.4% of valid |
| Qwen-Instruct 7B | Prompt engineering | 86.5% | 97.5% | 62.1% of valid |
| DeepSeek-R1-Distill-Llama-8B | LoRA fine-tuning | 100% | 100% | 81.5% of valid |

Format % is strict schema compliance (parseable structured output). Validity % is parseable AND contains actionable content. Accuracy counts cases where the generated Correction receives a human score of 0.8 or higher on a six-point rubric (0.0 = no match, 1.0 = perfect match).

Three things stand out.

**First, the raw DeepSeek-R1 model produces almost nothing usable.** Only 8% of outputs are valid, because reasoning models default to verbose chain-of-thought mode. When you ask DeepSeek-R1 for a structured correction, it thinks out loud at length and never arrives at a parseable output. This is not a knowledge failure — when the model does produce valid output, its 56% accuracy is reasonable. It is a schema compliance failure.

**Second, the instruction-tuned Qwen model solves the format problem but not the accuracy problem.** Qwen-Instruct achieves 97.5% validity — it follows output schemas reliably. But its accuracy is only 62.1%, compared to 64.4% for the prompt-engineered DeepSeek-R1 on the small subset where that model produces valid output. Instruction tuning gives you schema compliance; it cannot inject the domain-specific repair taxonomy.

**Third, LoRA fine-tuning resolves both problems simultaneously.** 100% format compliance and 81.5% accuracy — a 19.4 percentage point absolute improvement over the best prompt-only baseline. Nearly one in five cases that would be wrong with prompting is right with fine-tuning.

The mechanism is not subtle: 2 million real claims internalise the corpus of actual repair actions, part numbers, and procedural phrasing that distinguishes correct corrections from plausible-but-wrong ones. No prompt can replicate that.

---

## The LoRA configuration

The paper uses parameter-efficient fine-tuning via LoRA (Hu et al., 2021). The key hyperparameters:

```python
# LoRA configuration (Mo et al. 2026, Table 2)
lora_config = LoraConfig(
    r=32,                    # rank — higher than the common r=4/8/16
    lora_alpha=32,           # scaling factor; alpha/r = 1.0 (no additional damping)
    lora_dropout=0.0,        # no dropout — 2M training examples makes overfitting unlikely
    target_modules=[
        "q_proj", "v_proj",  # query and value projections, minimum config
        # optionally: "o_proj", "up_proj", "gate_proj", "down_proj"
    ],
    bias="none",
    task_type="CAUSAL_LM",
)
```

Training runs for a single epoch on the ~2M dataset with AdamW at learning rate 6×10⁻⁵, FP16 mixed precision, effective batch size 32 (per-device 8, gradient accumulation 4). The training loss is masked: computed only on the Correction tokens in the output, not on the Complaint/Cause prefix. This is standard causal language modelling with masked loss — the model is learning to generate corrections, not to reconstruct its own inputs.

Rank 32 is on the higher end for LoRA (common defaults are r=4 to r=16). The justification is the domain vocabulary. Automotive repair taxonomy — part codes, procedural sequences, torque specifications — is highly specific and low-frequency relative to general text. A higher rank preserves more expressive capacity for the adaptation. Alpha equal to rank gives a scaling factor of 1.0, meaning the LoRA updates are applied at full weight.

The practical benefit of LoRA here is compute cost. An 8B model with rank-32 LoRA has approximately 1.2M trainable parameters — well under 0.1% of the base model's parameter count. Fine-tuning 1.2M parameters on 2M examples in one epoch is feasible on a single A100 (80GB) in a few hours. Full fine-tuning of the 8B backbone would require multi-GPU infrastructure and significantly longer training times. The frozen backbone also means the general language capability of DeepSeek-R1-Distill is preserved — the adaptation is learning domain vocabulary and format, not rewriting general reasoning.

---

## Evaluation: why they did not use ROUGE

ROUGE and BLEU scores are standard in NLP but measure surface-level token overlap. For claims correction generation, two strings can be semantically identical and score near zero on ROUGE (different phrasing of the same repair action). The paper instead calibrates against human judgements.

The correlation analysis (Table 3 in the paper) compares automated metrics against human ratings on 200 cases:

| Metric | Chatterjee ξ | Spearman ρ |
|---|---|---|
| BERT cosine similarity | 0.692 | 0.733 |
| LLM-as-a-Judge | 0.606 | 0.724 |
| BLEURT (normalised) | 0.576 | 0.703 |

BERT cosine similarity is the best automated proxy for human judgement. The paper uses this to set a threshold (score ≥ 0.8 = acceptable), validates the threshold against human ratings, and then applies the threshold at scale. That is a sensible pipeline: ground the automated metric in human judgement, then use the automated metric for scalable evaluation.

The error analysis on 81 low-scoring cases is useful:

- Incorrect repair part: 28.4% of errors
- Surface form divergence (synonymous action, different wording): 23.4%
- Repair omission (correct but incomplete): 21.0%
- Granularity mismatch (too general or too specific): 19.8%
- Incorrect repair action: 7.4%

Only 7.4% of failures are outright wrong. The majority are stylistic or specificity issues — the model knows the right repair category but cannot always resolve the exact phrasing or granularity. That is an ontology problem as much as a model problem, and it is the kind of thing that improves with better training data curation rather than a fundamentally different architecture.

---

## The honest transfer to UK insurance

This is automotive warranty data from a US software provider (PCMI Corporation). It is not UK motor FNOL data. The vocabulary differs; the regulatory context differs; the claims taxonomy differs. We are not saying "deploy this on UK motor claims and expect 81.5% accuracy."

What we are saying is that the methodology transfers directly.

UK motor FNOL narratives have the same 3C structure. The customer complaint ("rear-end impact on M25, passenger complained of neck pain, third party admitted liability"), the cause (adjuster assessment of fault and injury severity), and the correction (settlement type, repair authorisation, total loss determination) map cleanly onto the same input/output schema. A model fine-tuned on UK motor claims with a similar LoRA setup would produce the same benefits: structured, parseable output that feeds directly into reserving and claims management workflows.

The data requirement is real. Mo et al. used 2 million records. You will not reproduce this on a dataset of 50,000 claims — fine-tuning on thin data produces models that overfit to the surface forms in the training set rather than learning the underlying repair logic. A UK insurer writing 500,000+ motor claims annually has enough data over a few years. A smaller insurer would need to consider whether consortium data (an industry dataset pooled across carriers) is feasible, or whether the technology is simply not yet cost-justified for their volume.

The second-order pricing connection is real but should not be overstated. Structured claims data produces better segmentation for IBNR reserving triangles — a motor claim correctly categorised as "electrical fault, battery replacement" enables finer development patterns than a single aggregate triangle. The structured output also creates categorical features that, if they predict future claim frequency or severity, can improve a pricing model. But that is two steps removed from the paper. The direct value is claims operations efficiency.

---

## UK regulatory posture

Fine-tuned LLMs making claims recommendations sit at the intersection of three regulatory frameworks that UK insurers need to think through before deployment.

**FCA Consumer Duty (PS22/9)** requires demonstrable fair outcomes. An AI system that systematically recommends lower-quality corrections for certain vehicle types, postcodes, or claim reporters creates outcome disparity that Consumer Duty directly targets. The duty is outcome-focused — the FCA wants evidence of good outcomes, not a particular governance process. That means you need outcome monitoring from day one, not bolted on after an audit finding.

**GDPR Article 22** covers automated decision-making with significant effects on individuals. Whether a claims correction recommendation qualifies as an automated decision depends on whether a human is materially reviewing it before the decision executes. A recommendation tool with adjuster review has a more defensible position than a fully autonomous settlement system. We think the right architecture for a first deployment is recommendation, not automation — the accuracy headroom (18.5% of cases still wrong) justifies human-in-the-loop even setting aside regulatory risk.

**SM&CR** requires a named senior manager accountable for the system. In practice this means the responsible person needs to understand what the model does, what its failure modes are, and how it is monitored. That requirement drives the practical specification: log input data, model version, confidence score (BERT cosine similarity to training distribution works as a proxy), decision output, and any human override. FCA AI audit trail guidance is expected by end of 2026; get the logging infrastructure right now rather than retrofitting it.

**Data residency** is a material advantage of local deployment. Running a fine-tuned 8B model on-premises or in a UK cloud region (AWS eu-west-2, Azure UK South) avoids the outsourcing concerns that arise under SYSC when claims data leaves the estate. A fine-tuned model that runs locally is not sending claims narratives to a US API endpoint. For UK insurers processing sensitive injury and third-party data, that is not a trivial distinction.

---

## How this connects to the embeddings work

We covered text embeddings on claims data in a [March 2026 post](/2026/03/26/nlp-text-embeddings-insurance-claims-pricing.html). The relationship between that work and this paper is a natural progression.

The embeddings approach takes claims text and produces dense vectors — opaque, continuous, useful as features in a pricing model or for similarity search. The claim automation approach goes further: it produces structured categorical output (correction type, parts category, repair procedure) from claims text. The structured output is more interpretable for reserving, more directly actionable for claims operations, and more auditable under Consumer Duty than an opaque embedding vector.

The pipeline sequence looks like this:

```
FNOL text
    ↓
[Embeddings approach]    → dense vector → pricing model feature
    ↓
[Claim automation]       → structured correction → claims workflow
                                                 → reserving triangle
                                                 → pricing feature (second-order)
```

They are not competing approaches. An insurer might use both: embeddings for pricing model features where semantic similarity is what you want, and fine-tuned structured output generation for claims workflow automation where parseable categorical fields are what downstream systems need.

---

## What would it cost to build this?

The paper does not report training time or GPU cost. Our rough estimate for a similar setup: one A100 80GB GPU running one epoch over 2M 512-token sequences in FP16 takes approximately 12–20 hours. At current UK cloud GPU pricing (£3–5/hour for an A100 spot instance on AWS or Azure), a single fine-tuning run costs roughly £60–100. The cost of repeating this quarterly as new claims data accumulates is negligible.

The dominant cost is not compute — it is data preparation. Labelling quality in training data matters more than GPU hours. Claims data in practice is messy: inconsistent coding, free-text fields with typos, correction fields that were entered by different adjusters with different conventions. Cleaning and standardising 2M records for training is weeks of data engineering work. The Mo et al. dataset came from PCMI's production warranty management system, which presumably has reasonably consistent taxonomy. Most insurers' legacy claims systems do not.

A medium-large UK insurer with a functioning data science team could build this. A small team without dedicated ML infrastructure should wait until hosted fine-tuning platforms (AWS Bedrock fine-tuning, Azure AI model customisation) mature further — the gap between API-hosted models and running your own LoRA checkpoint is closing.

---

## The CAS signal

Two active Casualty Actuarial Society RFPs validate that this is a funded, institutionally recognised problem.

The 2025 RFP ("Leveraging LLMs in Unstructured Claims Data", $40,000, deadline May 2025) covered exactly the Mo et al. use case: LLM-based conversion of unstructured claims data to categorical variables for reserving or ratemaking.

The 2026 RFP ("Adapting LLMs for P&C Actuarial Reasoning", $80,000, deadline April 27 2026 — currently open) explicitly lists LoRA and QLoRA as acceptable adaptation methods and covers pricing, reserving, capital modelling, and reinsurance. A team with the Mo et al. methodology adapted to P&C data and connected to actuarial workflows would be a strong candidate. The doubled budget (relative to 2025) signals that CAS expects proposals involving actual model training, not prompting exercises.

---

## The bottom line

Domain fine-tuning decisively outperforms prompt engineering for structured output generation from claims text. The Mo et al. result — 81.5% accuracy for LoRA fine-tuned versus 62.1% for the best prompt-only approach — is a 19-point absolute gap on a production-scale dataset. The mechanism is straightforward: 2 million domain-specific examples teach the model vocabulary and conventions that no prompt can replicate.

The technology is accessible. A fine-tuned 8B model runs on a single A100 GPU. LoRA keeps trainable parameters under 1.5M. Training cost is tens of pounds per run. The regulatory path is clear enough if you build recommendation rather than autonomous decision, maintain audit logs, and monitor outcomes from deployment.

The hard part is data — not compute. Claims data that is clean enough, consistently coded enough, and large enough to fine-tune on requires investment in data infrastructure that precedes any ML work. For insurers that have that infrastructure, this is real and buildable now.

For those who do not, the first step is not to fine-tune a model — it is to clean up the claims database.
