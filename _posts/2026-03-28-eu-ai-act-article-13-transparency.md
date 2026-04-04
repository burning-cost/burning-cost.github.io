---
layout: post
title: "EU AI Act Article 13: what transparency actually requires for a pricing model"
date: 2026-03-28
categories: [regulation]
tags: [eu-ai-act, regulation, model-governance, article-13, transparency, model-documentation, pra-ss123, shap, model-card, life-insurance, health-insurance, compliance]
description: "Article 13 of the EU AI Act is not about SHAP values. It is about deployer-facing documentation — what the underwriter or product team needs to interpret and use a pricing model correctly. We walk through what the Article actually requires, where PRA SS1/23 already covers the ground, and what is missing."
---

This is the third post in our EU AI Act series for insurance pricing teams. [Post 1]({{ site.baseurl }}{% post_url 2026-03-28-eu-ai-act-insurance-pricing-what-you-need-to-know %}) covered scope — who is in scope, which model types qualify as AI systems, and what the four-step decision sequence looks like. [Post 2](/2026/03/28/eu-ai-act-conformity-assessment-pricing-model/) covered the self-assessment process under Annex VI. This post is about Article 13 specifically.

Article 13 is one of the Articles that generates the most confusion, for a simple reason: people conflate "transparency" with "interpretability" or "explainability." They are different things, and the Act cares about a different one than most teams assume.

---

## What Article 13 actually says

The core obligation is in Article 13(1):

> *"High-risk AI systems shall be designed and developed in such a way as to ensure that their operation is sufficiently transparent to enable deployers to interpret a system's output and use it appropriately."*

"Deployers" in the Act's terminology are the organisations or individuals using the system in a professional context — for an in-house pricing model, that means the underwriting function, the product team, the distribution channel, the actuarial function reviewing outputs. The provider (the entity that built and deployed the model) is the insurer itself. Where an insurer is both provider and deployer — which is the normal position for in-house models — the Article 13 obligation is effectively the insurer's obligation to document the model well enough that non-builders can use it correctly.

Article 13(2) requires that high-risk AI systems be accompanied by *instructions for use* in an appropriate digital format. These must be "concise, complete, correct and clear" and "relevant, accessible and comprehensible to deployers." This is not the technical documentation of Article 11 — that document satisfies the regulator. The Article 13 document is operational: it satisfies the user.

Article 13(3) specifies the required contents. At minimum the instructions must include:

**(a)** Identity and contact details of the provider (for in-house models, the model owner and team).

**(b)** The characteristics, capabilities, and performance limitations of the system. This sub-paragraph is the substantive one. It requires, at minimum:

- The *intended purpose* — Article 13(3)(b)(i). Not a vague description: the specific use case, the lines of business, the types of risk, the decision the model feeds.
- The *level of accuracy, robustness and cybersecurity* against which the system was tested, and the *relevant accuracy metrics* — Article 13(3)(b)(ii). Including known or foreseeable circumstances that affect accuracy.
- Known or foreseeable circumstances that could lead to risks to health, safety, or fundamental rights — Article 13(3)(b)(iii).
- Technical capabilities to explain its outputs, *where applicable* — Article 13(3)(b)(iv). Note the caveat: where applicable. The Act does not require explainability tools for every model. It requires disclosure of whatever explainability tools exist.
- *Performance for specific persons or groups* — Article 13(3)(b)(v). This is the sub-paragraph that matters for actuaries. If your model performs materially differently for different age groups, health cohorts, or geographic segments, that difference must be disclosed.
- Input data specifications — Article 13(3)(b)(vi).
- Information enabling deployers to interpret the output and use it appropriately — Article 13(3)(b)(vii).

**(c)** Changes to the system that have been pre-determined and planned, and their effect on the system's performance.

**(d)** Human oversight measures — the technical measures that support human interpretation of outputs, cross-referencing Article 14.

**(e)** Computational and hardware resources needed, expected lifetime, maintenance schedule, and care measures.

Article 13(3)(f), in some versions of the text, also covers mechanisms for deployers to collect, store, and interpret logs per Article 12 — record-keeping linkage.

The transparency guidelines the Commission has committed to publish in Q2 2026 are expected to clarify interpretation of ambiguous provisions, but the core structure of Article 13 is settled text.

---

## Transparency is not interpretability

This is the distinction that matters most. The Act creates two separate obligations, and most commentary conflates them.

**Transparency** (Article 13) means: the deployer can understand what the model does, what inputs it requires, what it was tested against, where it underperforms, and when to question its output. This is a documentation obligation. It can be satisfied with well-written prose, tables, and performance summaries.

**Interpretability or explainability** (Article 13(3)(b)(iv)) is a *component* of transparency — but only "where applicable." The Act does not require every high-risk AI system to ship with SHAP values or individual-level explanations. It requires disclosure of whatever technical explanation capabilities *exist*. If you have SHAP, you must document that you have SHAP and how to use it. If you have nothing beyond global feature importance plots, you must document that limitation.

This matters practically: a team that has no per-prediction explanation tooling is not non-compliant with Article 13(3)(b)(iv), provided the instructions for use accurately state that fact. The risk is not non-disclosure of SHAP — it is failing to disclose the limitation.

There is a separate and more demanding explainability obligation under GDPR Article 22, which applies if pricing decisions are fully automated and have a legal or similarly significant effect on individuals. The GDPR obligation is to the *subject* of the decision (the policyholder). The Article 13 obligation is to the *deployer* (the underwriter, product team). These are different audiences, different documents, different legal bases.

Recital 47 of the Act makes the separation explicit: the transparency requirement is designed so that deployers — not affected individuals — can use the model appropriately. The individual rights question is handled through other instruments.

---

## What the Article 13 document actually looks like

The Article 13 document is an operational model fact sheet. Think of it as the answer to the question: *"what does the person using this model's output need to know?"*

For a life insurance pricing GBM, a compliant document covers:

**1. Identity and purpose**

Model name, version number, owner, effective date. Stated intended purpose in terms a product team can understand: "This model generates individual premium rates for term life new business with sum assured between £50,000 and £5 million, for applicants aged 18–75. It is not validated for use on group business, reinstatement pricing, or automatic acceptance limit setting."

**2. Accuracy statement and known limitations**

The actual performance metrics. Not "strong discriminatory power" — that is vendor language, not Article 13 content. The required form is specific: "Gini coefficient 0.44 (95% CI: 0.40–0.48) on out-of-time validation set covering policy years 2022–2023." Add the calibration: "Mean loss ratio predicted/observed ratio 0.97 overall. Known degradation: applicants aged 60–75 (Gini 0.31), applicants with BMI >35 (Gini 0.28). Model was not validated on applicants with more than two standard underwriting rated conditions — output in this segment should be treated with caution."

Article 13(3)(b)(ii) requires known and foreseeable circumstances affecting accuracy. This is a forward-looking requirement, not just a report on historical validation. If you expect model performance to degrade when macro conditions change materially (e.g. post-pandemic mortality patterns), that expectation belongs in the document.

**3. Performance for specific groups — Article 13(3)(b)(v)**

This is the sub-paragraph that most closely tracks the fairness disclosure debate. For insurance, the relevant groups are typically age cohorts (where data is thin at extremes), health status proxies (where model variables correlate with protected characteristics), and geographic segments (where postcode proxies socioeconomic status and indirectly other characteristics).

The disclosure does not require that performance be uniform across groups — that would be impossible given the actuarial relationship between age and mortality risk. It requires that performance *differences* be disclosed so deployers know where model outputs are less reliable. A compliant disclosure: "Model accuracy is materially lower for applicants aged under 25 due to limited training data exposure. In this segment, manual underwriting review is recommended for rated cases."

**4. Input data specifications**

The required features, acceptable ranges, how the model handles missing values, and what happens when inputs are outside the training distribution. This is the document that prevents a distribution team from feeding the model a corrupted application file and treating the output as valid.

**5. Explanation capabilities — Article 13(3)(b)(iv)**

Whatever you have. If the model is served via an API that returns SHAP values, document how to request and interpret them. If SHAP values are available only in the model team's tooling and not in the underwriter interface, say so — and say what alternative guidance is available for anomalous cases.

EIOPA-BoS-25-360 (paras 3.25–3.28) endorses SHAP and LIME for the explainability function, but notes their assumptions and limitations must be documented. The important limitation for GBMs: SHAP TreeExplainer values are exact for the model but reflect the model's learned representation, not causal relationships in the underlying risk. A SHAP value showing postcode as a top contributor tells the underwriter something real about the model; it does not tell them whether postcode is a legitimate risk differentiator or a proxy for a protected characteristic. That distinction belongs in the Article 13 document, not the SHAP output.

**6. Human oversight measures — Article 13(3)(d)**

When should an underwriter override the model? The Article 13 document is where you state this explicitly. Minimum content: the anomaly thresholds that should trigger manual review, the procedure for logging overrides, who has authority to suspend automated pricing, and what training the oversight persons have received. This cross-references Article 14 obligations, but from the deployer's perspective — what do they need to know to exercise oversight correctly?

**7. Maintenance schedule**

Expected model lifetime, next planned retraining, the monitoring metrics and thresholds that would trigger out-of-cycle retraining, and who is responsible for the maintenance decision. Article 13(3)(e) requires expected lifetime and maintenance measures. A pricing team retraining annually should document the retraining cycle and the conditions that would accelerate it.

---

## SHAP and model cards: are they sufficient?

Not on their own, for two reasons.

First, SHAP solves a different problem. SHAP provides feature attributions for individual predictions or mean attributions across a dataset. It does not address most of what Article 13(3) requires: the intended purpose declaration, the accuracy metrics by sub-population, the input specification, the human oversight procedure, the maintenance schedule. SHAP is, at most, the content that satisfies Article 13(3)(b)(iv). The rest of the document requires deliberate authoring.

Second, a standard model card (in the Mitchell et al. 2019 sense) covers some of the Article 13 ground but misses the operational emphasis. A model card answers "what did we build and how does it perform?" The Article 13 document must also answer "how do you use this correctly?" — which includes human oversight procedures, input handling instructions, and maintenance commitments. The deployer orientation is the distinguishing feature.

What satisfies Article 13 is a model card-plus: all the standard model card content (intended use, out-of-scope uses, performance metrics, evaluation data, ethical considerations), supplemented by the operational sections that most model cards omit — input specifications, override procedures, maintenance schedule, and explicit sub-population performance disclosure.

The EIOPA Annex I record-keeping template from the Final Opinion is a useful reference: it covers reasons for using AI, IT integration, staff roles, data collection (including ground truth construction and bias removal), data preparation (variables, domain ranges, feature engineering), technical choices (algorithm rationale, library versions), code and data governance, and model performance KPIs with satisfactory performance levels and review frequency. That template, completed properly, would produce an Article 13-adequate document for lower-risk AI under the EIOPA framework. For high-risk AI, it covers approximately 70% of the Article 13 requirement — the human oversight and maintenance sections need explicit addition.

---

## How PRA SS1/23 maps to Article 13

PRA SS1/23 applies to banks, not insurers — but many UK pricing teams have adopted equivalent SS1/23-aligned governance practices voluntarily. Those teams already have documentation obligations that cover substantial Article 13 ground. The mapping is close enough that most of what an Article 13 document requires can be assembled from existing SS1/23 artefacts, with targeted additions.

| Article 13 requirement | SS1/23 equivalent | Gap |
|---|---|---|
| Provider identity, model version | Model inventory entry | Usually present |
| Intended purpose, scope limitations | Model use statement in validation report | Usually present but often vague |
| Accuracy metrics and validation results | Validation report performance section | Usually present; needs sub-population breakdown |
| Known limitations and failure modes | Limitations section of validation report | Present but often not quantified |
| Performance for specific groups | Not explicitly required by SS1/23 | **Gap: typically absent** |
| Input data specifications | Data governance section | Partially present |
| Explanation capabilities | Not explicitly required | **Gap: absent in most teams** |
| Human oversight procedures | Model use policy, escalation procedures | Partially present; not model-specific |
| Maintenance schedule | Periodic review trigger in model policy | Present but often generic |

The two gaps that appear consistently in SS1/23-compliant firms are the sub-population performance disclosure and the explanation capabilities section. These are Article 13(3)(b)(iv) and (v) respectively. They require deliberate work because SS1/23 does not drive teams to quantify performance differences by demographic-adjacent segments or to document what explanation tools exist and what their limitations are.

---

## Python: generating an Article 13 document from a pricing model

The code below generates an Article 13-structured documentation template from a fitted scikit-learn model. It does not replace the human authoring required for the narrative sections — but it automates the extractable sections and flags the gaps that require manual input.

```python
from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score


@dataclass
class Article13Document:
    """
    Structured Article 13 transparency document for a high-risk AI system
    under Regulation (EU) 2024/1689.

    Fields map to Article 13(3)(a)-(e).
    """

    # Art 13(3)(a)
    provider_name: str
    provider_contact: str
    model_name: str
    model_version: str
    document_date: date

    # Art 13(3)(b)(i)
    intended_purpose: str
    out_of_scope_uses: list[str]

    # Art 13(3)(b)(ii) — populated by compute_accuracy()
    accuracy_metrics: dict[str, Any] = field(default_factory=dict)
    known_accuracy_limitations: list[str] = field(default_factory=list)

    # Art 13(3)(b)(iii)
    known_risks: list[str] = field(default_factory=list)

    # Art 13(3)(b)(iv)
    explanation_tools: str = ""

    # Art 13(3)(b)(v) — populated by compute_subgroup_performance()
    subgroup_performance: dict[str, Any] = field(default_factory=dict)

    # Art 13(3)(b)(vi)
    input_features: list[dict[str, str]] = field(default_factory=list)

    # Art 13(3)(b)(vii)
    output_interpretation_guide: str = ""

    # Art 13(3)(c)
    planned_changes: list[str] = field(default_factory=list)

    # Art 13(3)(d)
    human_oversight_measures: str = ""
    override_procedure: str = ""
    anomaly_thresholds: dict[str, float] = field(default_factory=dict)

    # Art 13(3)(e)
    expected_lifetime_months: int = 0
    next_retraining_date: date | None = None
    retraining_triggers: list[str] = field(default_factory=list)
    monitoring_metrics: list[str] = field(default_factory=list)

    def compute_accuracy(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        label: str = "out-of-time test set",
    ) -> None:
        """
        Compute and store accuracy metrics. Art 13(3)(b)(ii) requires the level
        of accuracy, the relevant metrics, and the conditions under which tested.
        """
        y_pred = model.predict_proba(X_test)[:, 1]
        gini = 2 * roc_auc_score(y_test, y_pred) - 1

        # Bootstrap CI on Gini
        rng = np.random.default_rng(42)
        bootstrap_ginis = []
        n = len(y_test)
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_test.iloc[idx])) < 2:
                continue
            g = 2 * roc_auc_score(y_test.iloc[idx], y_pred[idx]) - 1
            bootstrap_ginis.append(g)

        ci_low = float(np.percentile(bootstrap_ginis, 2.5))
        ci_high = float(np.percentile(bootstrap_ginis, 97.5))

        self.accuracy_metrics = {
            "dataset": label,
            "n_records": int(n),
            "gini_coefficient": round(gini, 4),
            "gini_95ci": [round(ci_low, 4), round(ci_high, 4)],
            "auc_roc": round(roc_auc_score(y_test, y_pred), 4),
            "prevalence": round(float(y_test.mean()), 4),
        }

    def compute_subgroup_performance(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        group_col: str,
        min_group_size: int = 100,
    ) -> None:
        """
        Compute per-group Gini. Art 13(3)(b)(v) requires performance disclosure
        for specific persons or groups where this is applicable.
        """
        results = {}
        y_pred = model.predict_proba(X_test)[:, 1]

        for group_val, idx in X_test.groupby(group_col).groups.items():
            if len(idx) < min_group_size:
                results[str(group_val)] = {
                    "n": len(idx),
                    "note": f"Insufficient data for reliable estimate (n={len(idx)} < {min_group_size})",
                }
                continue

            y_g = y_test.iloc[X_test.index.get_indexer(idx)]
            p_g = y_pred[X_test.index.get_indexer(idx)]

            if len(np.unique(y_g)) < 2:
                results[str(group_val)] = {
                    "n": len(idx),
                    "note": "Single outcome class in group — Gini undefined",
                }
                continue

            gini_g = round(2 * roc_auc_score(y_g, p_g) - 1, 4)
            results[str(group_val)] = {"n": len(idx), "gini": gini_g}

        self.subgroup_performance[group_col] = results

    def flag_gaps(self) -> list[str]:
        """
        Return list of unfilled required sections. Use this to track what
        requires manual authoring before the document is compliant.
        """
        gaps = []

        if not self.intended_purpose:
            gaps.append("Art 13(3)(b)(i): intended_purpose not set")
        if not self.accuracy_metrics:
            gaps.append("Art 13(3)(b)(ii): accuracy_metrics not computed — call compute_accuracy()")
        if not self.known_accuracy_limitations:
            gaps.append("Art 13(3)(b)(ii): known_accuracy_limitations not documented")
        if not self.known_risks:
            gaps.append("Art 13(3)(b)(iii): known_risks not documented")
        if not self.explanation_tools:
            gaps.append("Art 13(3)(b)(iv): explanation_tools not documented")
        if not self.subgroup_performance:
            gaps.append("Art 13(3)(b)(v): subgroup_performance not computed — call compute_subgroup_performance()")
        if not self.input_features:
            gaps.append("Art 13(3)(b)(vi): input_features not specified")
        if not self.output_interpretation_guide:
            gaps.append("Art 13(3)(b)(vii): output_interpretation_guide not authored")
        if not self.human_oversight_measures:
            gaps.append("Art 13(3)(d): human_oversight_measures not documented")
        if not self.override_procedure:
            gaps.append("Art 13(3)(d): override_procedure not documented")
        if self.expected_lifetime_months == 0:
            gaps.append("Art 13(3)(e): expected_lifetime_months not set")
        if not self.retraining_triggers:
            gaps.append("Art 13(3)(e): retraining_triggers not documented")

        return gaps

    def to_markdown(self) -> str:
        """Render Article 13 document as structured Markdown."""
        lines = [
            f"# Article 13 Transparency Document",
            f"**Regulation (EU) 2024/1689 — High-Risk AI System**",
            f"",
            f"| Field | Value |",
            f"|---|---|",
            f"| Model | {self.model_name} v{self.model_version} |",
            f"| Provider | {self.provider_name} |",
            f"| Contact | {self.provider_contact} |",
            f"| Document date | {self.document_date} |",
            f"",
            f"---",
            f"",
            f"## Art 13(3)(b)(i) — Intended purpose",
            f"",
            textwrap.fill(self.intended_purpose, width=80),
            f"",
            f"**Out of scope:**",
        ]

        for oos in self.out_of_scope_uses:
            lines.append(f"- {oos}")

        lines += [
            f"",
            f"---",
            f"",
            f"## Art 13(3)(b)(ii) — Accuracy, robustness and known limitations",
            f"",
            f"```json",
            json.dumps(self.accuracy_metrics, indent=2),
            f"```",
            f"",
            f"**Known accuracy limitations:**",
        ]

        for lim in self.known_accuracy_limitations:
            lines.append(f"- {lim}")

        lines += [
            f"",
            f"---",
            f"",
            f"## Art 13(3)(b)(iii) — Known risks",
            f"",
        ]

        for risk in self.known_risks:
            lines.append(f"- {risk}")

        lines += [
            f"",
            f"---",
            f"",
            f"## Art 13(3)(b)(iv) — Explanation capabilities",
            f"",
            textwrap.fill(self.explanation_tools or "_Not documented_", width=80),
            f"",
            f"---",
            f"",
            f"## Art 13(3)(b)(v) — Performance for specific groups",
            f"",
        ]

        if self.subgroup_performance:
            for col, groups in self.subgroup_performance.items():
                lines.append(f"### Breakdown by: `{col}`")
                lines.append(f"")
                lines.append(f"| Group | N | Gini | Note |")
                lines.append(f"|---|---|---|---|")
                for grp, metrics in groups.items():
                    gini = metrics.get("gini", "—")
                    note = metrics.get("note", "")
                    n = metrics.get("n", "—")
                    lines.append(f"| {grp} | {n} | {gini} | {note} |")
                lines.append(f"")
        else:
            lines.append(f"_Not computed — call compute_subgroup_performance()_")
            lines.append(f"")

        lines += [
            f"---",
            f"",
            f"## Art 13(3)(b)(vi) — Input data specifications",
            f"",
            f"| Feature | Type | Acceptable range | Missing value handling |",
            f"|---|---|---|---|",
        ]

        for feat in self.input_features:
            lines.append(
                f"| {feat.get('name','?')} | {feat.get('type','?')} "
                f"| {feat.get('range','?')} | {feat.get('missing','?')} |"
            )

        lines += [
            f"",
            f"---",
            f"",
            f"## Art 13(3)(b)(vii) — Output interpretation",
            f"",
            textwrap.fill(self.output_interpretation_guide or "_Not authored_", width=80),
            f"",
            f"---",
            f"",
            f"## Art 13(3)(d) — Human oversight",
            f"",
            textwrap.fill(self.human_oversight_measures or "_Not documented_", width=80),
            f"",
            f"**Override procedure:** {self.override_procedure or '_Not documented_'}",
            f"",
        ]

        if self.anomaly_thresholds:
            lines += [
                f"**Anomaly thresholds triggering manual review:**",
                f"",
            ]
            for metric, threshold in self.anomaly_thresholds.items():
                lines.append(f"- `{metric}`: {threshold}")
            lines.append(f"")

        lines += [
            f"---",
            f"",
            f"## Art 13(3)(e) — Maintenance",
            f"",
            f"- Expected lifetime: {self.expected_lifetime_months} months",
            f"- Next planned retraining: {self.next_retraining_date or 'Not scheduled'}",
            f"",
            f"**Triggers for out-of-cycle retraining:**",
        ]

        for trigger in self.retraining_triggers:
            lines.append(f"- {trigger}")

        lines += [
            f"",
            f"**Monitoring metrics in production:**",
        ]

        for metric in self.monitoring_metrics:
            lines.append(f"- {metric}")

        gaps = self.flag_gaps()
        if gaps:
            lines += [
                f"",
                f"---",
                f"",
                f"## COMPLIANCE GAPS — sections requiring manual completion",
                f"",
            ]
            for gap in gaps:
                lines.append(f"- {gap}")

        return "\n".join(lines)
```

Usage — assembling the document for a life pricing GBM:

```python
from datetime import date

doc = Article13Document(
    provider_name="Acme Life Insurance plc",
    provider_contact="pricing-team@acme.example.com",
    model_name="TermLife-GBM-v3",
    model_version="3.2.1",
    document_date=date(2026, 3, 28),
    intended_purpose=(
        "GBM model generating individual premium rates for term life new business. "
        "Applicable to applicants aged 18–75, sum assured £50,000–£5,000,000. "
        "Standard and non-standard lives. Not validated for group business, "
        "reinstatement pricing, or automatic acceptance limit setting."
    ),
    out_of_scope_uses=[
        "Group life scheme rating",
        "Claims reserving or IBNR estimation",
        "Reinsurance pricing",
        "Setting maximum benefit limits or automatic acceptance limits",
        "Any use outside declared feature ranges",
    ],
    known_accuracy_limitations=[
        "Gini coefficient 0.31 for applicants aged 60-75 vs 0.44 overall — thin "
        "training data for older ages. Manual review recommended for rated cases.",
        "Gini coefficient 0.28 for BMI > 35 segment. Model was not trained on "
        "sufficient impaired lives with high BMI. Treat outputs with elevated caution.",
        "Model not validated post-COVID mortality shock. Performance may degrade "
        "if excess mortality patterns persist materially into future experience.",
        "Postcode feature may act as a proxy for socioeconomic status. "
        "Sub-population performance by deprivation quintile has not been validated.",
    ],
    known_risks=[
        "Proxy discrimination: postcode variable correlates with ethnic composition "
        "of areas. Risk that model infers protected characteristics indirectly.",
        "Distribution shift: model trained on 2018-2023 experience. Post-pandemic "
        "mortality trends not fully represented.",
        "Automation bias: underwriters may over-rely on model output without "
        "applying case-specific judgment for impaired lives.",
        "Feedback loop: if rated-out cases are systematically excluded from "
        "training data, model may learn biased risk boundaries over time.",
    ],
    explanation_tools=(
        "SHAP TreeExplainer values are computed per prediction and returned via "
        "the model API. Global SHAP summary plots are available in the model "
        "documentation portal. Note: SHAP values reflect the model's learned "
        "representation and do not imply causality. Underwriters should treat "
        "high postcode SHAP values as a flag for manual review, not as evidence "
        "that geography is the causal risk driver."
    ),
    input_features=[
        {"name": "age_at_entry", "type": "int", "range": "18-75", "missing": "Reject — required field"},
        {"name": "smoker_status", "type": "binary", "range": "0/1", "missing": "Impute as non-smoker with flag"},
        {"name": "sum_assured", "type": "float", "range": "50000-5000000", "missing": "Reject — required field"},
        {"name": "bmi", "type": "float", "range": "15-55", "missing": "Use mean imputation; flag for review if >40"},
        {"name": "postcode_sector", "type": "str", "range": "Valid UK postcode sector", "missing": "Use national average; flag"},
    ],
    output_interpretation_guide=(
        "Model returns a risk multiplier (base rate = 1.0). Multipliers above 2.5 "
        "should trigger manual underwriting review before rate is quoted. "
        "Multipliers below 0.6 are unusual for this model and should also be "
        "reviewed. The model does not return a probability of claim directly — "
        "the risk multiplier must be applied to the office's current base rates. "
        "Do not interpret small differences in multiplier (< 0.05) as materially "
        "different risk — uncertainty at this precision is within model noise."
    ),
    planned_changes=[
        "v3.3 planned Q4 2026: retraining on 2024-2025 experience to capture "
        "post-pandemic normalisation. Will trigger new Article 13 document.",
        "v4.0 planned 2027: evaluation of neural architecture alternative. "
        "Substantial modification if adopted — new conformity assessment required.",
    ],
    human_oversight_measures=(
        "All multipliers above 2.5 are automatically held for senior underwriter "
        "review. Underwriters receive mandatory training on automation bias annually. "
        "The pricing system logs all overrides with reason codes. The actuarial "
        "function reviews override rates quarterly. If override rates exceed 15% "
        "of referred cases, model performance is investigated within 30 days."
    ),
    override_procedure=(
        "Underwriter enters override reason code in system; model output is "
        "preserved alongside override decision for monitoring. Override decisions "
        "escalated to head of underwriting for sum assured > £1,000,000."
    ),
    anomaly_thresholds={
        "risk_multiplier_upper": 2.5,
        "risk_multiplier_lower": 0.6,
        "monthly_override_rate_pct": 15.0,
        "psi_vs_training": 0.25,
    },
    expected_lifetime_months=18,
    next_retraining_date=date(2027, 3, 1),
    retraining_triggers=[
        "PSI > 0.25 on any top-5 feature (monthly monitoring)",
        "Portfolio Gini drops below 0.38 on rolling 6-month experience",
        "Actual/expected mortality ratio outside 90%-110% for two consecutive quarters",
        "Material change in product design or target market",
    ],
    monitoring_metrics=[
        "Monthly PSI on age, sum assured, BMI, smoker rate, postcode distribution",
        "Rolling 6-month Gini on new business with first-year claims",
        "Actual/expected claims count by risk band (quarterly)",
        "Override rate by underwriting team (quarterly)",
        "Feature SHAP stability — mean absolute change in global importance vs training baseline",
    ],
)

# Compute accuracy from your test set
doc.compute_accuracy(model, X_test, y_test, label="out-of-time 2022-2023")

# Compute sub-population performance
doc.compute_subgroup_performance(model, X_test, y_test, group_col="age_band")

# Check what is missing before filing
gaps = doc.flag_gaps()
for gap in gaps:
    print(f"GAP: {gap}")

# Render the document
markdown_output = doc.to_markdown()
with open("article13_termlife_gbm_v3.md", "w") as f:
    f.write(markdown_output)
```

The `flag_gaps()` method is the practical tool: it tells you, systematically, which Article 13(3) sub-paragraphs remain unfilled before you file the conformity pack. A document with no gaps from `flag_gaps()` still needs human authoring review — some fields could be technically present but poorly written. But an empty gap list is the minimum bar.

---

## What UK pricing teams should do now

The EU AI Act's Article 13 requirements apply from 2 August 2026 for high-risk AI — which means life and health pricing GBMs, neural networks, and model classes that qualify as AI systems under Article 3(1). That is four months from now.

For UK teams, the position depends on exposure:

**UK-only life insurer, no EU policyholders:** EU AI Act does not apply. But good model governance practice — and for banks, PRA SS1/23 specifically — requires model documentation that covers much of the same ground. Insurers are not subject to SS1/23 directly; the Solvency II Articles 120–126 documentation requirements produce substantially similar artefacts. The sub-population performance disclosure (Article 13(3)(b)(v)) and the explicit explanation capability statement (Article 13(3)(b)(iv)) are gaps in most SS1/23 packs. Worth closing regardless.

**UK insurer with EEA subsidiary or cross-border EU business:** Article 13 documentation is a legal requirement for the models feeding EU policyholder decisions. Four months to deadline. Start with the gap analysis against existing validation reports. The `flag_gaps()` output from the code above is a reasonable starting checklist.

**Lloyd's managing agents:** Brussels-written business is within EU jurisdiction. Pricing models for Lloyd's Brussels syndicates are in scope if they are GBM/NN models for life or health lines. Market-level coordination on documentation standards would be useful; we are not aware of any coordinated approach as of March 2026.

**General insurance teams — motor, property, commercial lines:** Not in scope for Article 13 high-risk AI obligations. But the EIOPA Final Opinion (EIOPA-BoS-25-360) applies the same governance logic — including the transparency and explainability section — to non-high-risk insurance AI. The spirit of Article 13 is what good model documentation looks like. Building it for motor GBMs is not over-compliance; it is the right level of documentation for a model driving pricing decisions for millions of policyholders.

The Commission's Q2 2026 transparency guidelines will add interpretive clarity on some ambiguous provisions — particularly what "sufficiently transparent" means in practice and whether statistical explanation methods (segment-level SHAP) satisfy Article 13(3)(b)(iv) where individual-level explanation is technically difficult. We will update this post when those guidelines are published.

---

*Post 1 in this series: [EU AI Act and Insurance Pricing — What You Actually Need to Know]({{ site.baseurl }}{% post_url 2026-03-28-eu-ai-act-insurance-pricing-what-you-need-to-know %}). Post 2: [EU AI Act Conformity Assessment for Pricing Models](/2026/03/28/eu-ai-act-conformity-assessment-pricing-model/). Primary sources: Regulation (EU) 2024/1689, OJ 12.7.2024; EIOPA-BoS-25-360, August 2025.*
