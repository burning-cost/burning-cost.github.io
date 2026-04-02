---
layout: post
title: "Can LLMs Pass Their Insurance Exams? The Wrong Question for Pricing Teams"
date: 2026-04-02
categories: [techniques, llm, pricing, actuarial, governance]
tags: [llm, rag, retrieval-augmented-generation, benchmark, insurance-advisory, compliance, consumer-duty, fca, solvency-ii, governance, actuarial-judgment, beauchemin-khoury, arXiv-2603-07825]
description: "Beauchemin & Khoury (arXiv:2603.07825) benchmark 51 LLMs on Quebec insurance regulatory certification questions. Passing insurance exams is the wrong success metric for pricing teams. The RAG finding is worth paying attention to — but not for the reason the paper emphasises."
math: false
author: burning-cost
---

A paper from Beauchemin and Khoury (arXiv:2603.07825, March 2026), presented at ICLR 2026's Financial AI workshop, introduces AEPC-QA: 807 multiple-choice questions drawn from Quebec insurance regulatory certification handbooks, benchmarked against 51 LLMs under both closed-book and retrieval-augmented generation conditions. The headline result — that RAG boosts weak models by more than 35 percentage points — has been picked up as evidence that LLMs are becoming viable insurance advisors.

They are not. And the benchmark tests the wrong thing to determine whether they are.

---

## What the benchmark actually tests

AEPC-QA is a certification knowledge test. The questions come from Quebec insurance regulatory handbooks — the material a new broker or financial advisor would study to pass their licensing exam. Think product definitions, regulatory obligations, disclosure requirements, and coverage mechanics at a general level. This is regulatory memorisation, not actuarial judgment.

There is a meaningful difference between those two things. A pricing actuary's job involves statistical model selection, distributional assumptions, credibility weighting, exposure measurement, and translating uncertain empirical evidence into rate decisions. None of this appears in a certification handbook, because none of it is what a broker exam tests.

The benchmark is also Quebec-specific in ways that matter. Quebec operates under the Civil Code and has its own insurance regulatory framework under the Autorité des marchés financiers. The AMF's certification requirements, the policy wordings tested, and the regulatory nuances differ meaningfully from the UK market. A model that tops the AEPC-QA leaderboard is not thereby qualified to advise on UK GI products under FCA Consumer Duty, or to populate a Solvency II ORSA narrative, or to answer questions about Lloyd's market participation. Separate benchmarks using CII qualification material and FCA regulatory text would need to be built and tested before drawing any conclusions about UK applicability.

---

## The closed-book finding

The paper establishes that closed-book LLM performance on insurance regulatory questions is, for most models, uncomfortably low. That 35-percentage-point gap when moving from closed-book to RAG is not primarily a story about how much RAG helps. It is a story about how unreliable closed-book LLMs are for insurance compliance questions.

A closed-book model getting 40% of regulatory questions wrong is not a tool you give to customers for product guidance. It is not a tool you give to compliance teams to draft regulatory submissions. The model does not know what it does not know, and in insurance, the gap between a plausible-sounding answer and a technically correct one can be the difference between a valid claim being paid and a coverage dispute.

This is the finding the paper should lead with, but does not, because it reflects poorly on the entire premise of using LLMs for advisory.

---

## The RAG finding, correctly understood

RAG works by retrieving relevant passages from a document corpus and injecting them into the model's context before it answers. The mechanism is simple: the model is no longer relying on its training data, which may be stale, geographically mismatched, or simply absent for specialised regulatory content. Instead, it is reading the relevant handbook page before answering.

The 35-percentage-point improvement tells you that, given the right passage in context, the model can extract the correct answer reasonably well. This is useful, but it is a retrieval task as much as a reasoning task.

The counter-intuitive finding — that RAG degrades strong models' performance — is practically significant. The likely explanation is that strong models have internalized enough insurance knowledge during training that injecting retrieved passages introduces conflicting signals or distracts from patterns the model has already learned. This is not a universal result. It depends on retrieval quality, chunk size, and how the retrieved content relates to what the model already knows.

For UK pricing teams, we think the correct reading of the RAG finding is this: LLMs with well-curated retrieval corpora are reasonable tools for governance documentation tasks where the answer is a function of retrieving and summarising existing text. This includes things like drafting explanations of model decisions in language appropriate for a regulatory submission, answering questions about what a specific Solvency II article requires, or summarising what your internal model documentation says about a particular risk factor. These are tasks where the source material exists and the requirement is coherent, accurate prose — not tasks that require actuarial judgment about evidence that does not yet exist.

---

## The Consumer Duty analogy

The paper frames Quebec's Bill 141 — which accelerated insurance digitisation without replacing human advisor judgment — as creating an "advice gap." The FCA's Consumer Duty has a structurally similar dynamic. The 2023 Consumer Duty rules increased the standard of care required for retail insurance customers while simultaneously not providing the industry with tools to scale that care digitally.

There is a real business problem here. The advice gap is genuine. But the conclusion that LLMs are the solution to the advice gap does not follow from a benchmark showing that RAG-augmented models perform better on certification questions. Passing a certification exam was not the bottleneck in human advisor quality. The bottleneck was time, cost, and availability of qualified humans. An LLM that passes certification questions at 80% accuracy still needs a human backstop for the 20%, and if that human backstop is needed, you have not solved the availability problem.

We think the honest version of the LLM insurance advisor story is: they are useful for information retrieval at scale, as a first-pass triage layer directing customers to the right product category, and as a documentation assistant for compliance teams. None of these are trivial, and all of them create real value. But they are not substitutes for judgment in coverage disputes, pricing decisions, or advice on complex commercial risks.

---

## Prior coverage

We have covered related LLM-in-insurance papers before. Balona's ActuaryGPT (KB 3899, 2024) showed GPT-4 passing actuarial exam questions — a different test, professional examination rather than regulatory certification, but the same underlying question about what exam-passing proves. The CAS's RFP work on LLM for unstructured claims data (KB 4840) and the LLM feature engineering papers (KB 4671-4673) are closer to pricing practice. This paper is distinct: advisory and compliance benchmarking, not claims processing or feature engineering.

---

## Our verdict

The benchmark is a reasonable contribution to understanding what LLMs currently know about insurance regulatory content in a specific jurisdiction. The AEPC-QA dataset is a real public resource and the 51-model comparison is useful empirical work.

For UK pricing actuaries, the practical takeaway is narrow: closed-book LLMs should not be trusted for compliance-sensitive tasks, and RAG with a curated regulatory document corpus is probably your best current option if you want LLM assistance with governance documentation. For anything involving pricing methodology, model validation, or actuarial judgment, the benchmark's results are irrelevant — not because LLMs are useless, but because the benchmark does not test those capabilities.

The more useful benchmark would test whether LLMs can explain the logic of a pricing model's output in language a non-specialist can understand, or flag inconsistencies between a model assumption document and the actual model specification. Nobody has built that benchmark yet. When they do, we will cover it.
