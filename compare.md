---
layout: page
title: "Open-Source Alternatives to Commercial Insurance Pricing Platforms"
description: "Pricing actuaries comparing Emblem, Radar, Akur8, or DataRobot to open-source Python tools. Burning Cost covers the full pricing workflow: GLMs, GBMs, rate optimisation, validation, and regulatory compliance. Free, MIT-licensed, UK-specific."
permalink: /compare/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "Is there a free alternative to Emblem for insurance pricing?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes. Burning Cost provides free, open-source Python libraries that cover what Emblem does for GLM pricing and more. shap-relativities extracts multiplicative factor tables from GBMs in the same format as exp(beta) from a GLM. insurance-cv provides temporally-correct walk-forward cross-validation with IBNR buffers. insurance-interactions detects GLM interaction effects. All libraries are MIT-licensed and available on PyPI. The trade-off is that you get a code-first workflow rather than a GUI — but you also get full transparency, version control, and no licence fees."
      }
    },
    {
      "@type": "Question",
      "name": "Can I use Python for insurance pricing instead of Emblem or Radar?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes, and a growing number of UK pricing teams are doing exactly this. Python with statsmodels or scikit-learn covers GLMs and GBMs. The gap has historically been in the actuarial-specific tooling: walk-forward cross-validation that respects IBNR, SHAP-based factor tables, constrained rate optimisation, PRA SS1/23 validation reports, and FCA Consumer Duty fairness auditing. Burning Cost fills those gaps with 28 focused Python libraries. Databricks provides the compute environment most UK insurers are already using."
      }
    },
    {
      "@type": "Question",
      "name": "What open-source tools exist for actuarial pricing?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Burning Cost publishes 28 open-source Python libraries for UK personal lines pricing. They cover: temporal cross-validation (insurance-cv), GBM factor tables (shap-relativities), prediction intervals (insurance-conformal, insurance-quantile), rate optimisation (rate-optimiser, insurance-optimise), causal inference (insurance-causal, insurance-elasticity), spatial rating (insurance-spatial), fairness auditing (insurance-fairness), model validation (insurance-validation), model monitoring (insurance-monitoring), and model governance (insurance-mrm). The chainladder-python project covers reserving. Meaningful open-source tooling for commercial pricing and rate filing is less developed."
      }
    },
    {
      "@type": "Question",
      "name": "Is there a free alternative to Akur8?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Akur8 focuses on automated GLM and GBM model building with a GUI for actuarial review. Burning Cost covers similar modelling territory in Python: shap-relativities for GBM factor extraction, insurance-anam for interpretable neural models, insurance-interactions for interaction detection, and bayesian-pricing for thin-segment modelling. You do not get a hosted GUI — you work in notebooks or scripts. The methodologies are comparable; the delivery mechanism is different."
      }
    },
    {
      "@type": "Question",
      "name": "What is the best Python library for insurance GLM pricing?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "For GLMs, statsmodels and scikit-learn (TweedieRegressor) are the standard base. Burning Cost adds the actuarial layer: insurance-cv for correct cross-validation, shap-relativities for GBM factor tables that match GLM output formats, insurance-interactions for interaction detection, credibility for thin-cell blending, and insurance-validation for PRA SS1/23 model reports. Together these give you a complete GLM and GBM pricing workflow in Python."
      }
    }
  ]
}
</script>

This page exists because pricing actuaries searching for open-source insurance pricing tools deserve a straight answer, not marketing.

Burning Cost is on the forefront of machine learning and data science research in UK personal lines insurance. We help teams adopt best practice, best-in-class tooling, and Databricks — 28 open-source Python libraries covering the full pricing workflow. We are not trying to compete with Emblem, Radar, Akur8, or DataRobot. Those tools have real strengths: polished UIs, enterprise support contracts, integration with downstream systems, and regulatory track records with insurers who do not want to maintain Python infrastructure.

What we offer is different: research-backed methodology, transparent implementations, version-controllable outputs, and specific focus on UK regulatory requirements. If you are a pricing team working in Python or Databricks, Burning Cost covers the actuarial gaps that general ML libraries do not.

If you need a hosted GUI, enterprise support, or a system that non-technical pricing managers can use without code, the commercial platforms are probably right for you. Both things can be true.

---

## What the commercial platforms do well

Before the comparison table, the honest version of what you are trading away if you go open-source:

**Emblem (WTW):** Mature GLM platform with decades of actuarial workflow built in. The factor review UI, one-way and two-way analyses, and the signed-off model export workflow are genuinely good. Many UK insurers have 10+ years of Emblem models in production. Migration is not trivial.

**Radar (Earnix):** Strong on the commercial optimisation side. Rate change simulation and price optimisation tooling that connects to production rating systems. Enterprise support and integration with downstream workflow systems.

**Akur8:** Machine learning model building with a GUI that lets actuaries interact with GBM outputs without writing code. Good for teams that want GBM predictive power without committing to a Python workflow. Growing adoption in UK and European markets.

**DataRobot:** General AutoML platform used by some insurers for pricing. Broad model coverage, explainability tools, and MLOps infrastructure. Not insurance-specific, but the enterprise deployment capabilities are mature.

None of these are bad tools. The question is whether the licence cost, vendor dependency, and opaque methodology are the right trade for your team.

---

## Feature comparison

The table below maps pricing workflow areas to Burning Cost libraries and notes what commercial platforms typically offer. For commercial tools, we describe capabilities as they are generally understood — we do not have access to their current feature sets, pricing, or implementation details.

| Feature area | Burning Cost (open-source) | Commercial platforms (typically) |
|---|---|---|
| **GLM modelling** | statsmodels, scikit-learn TweedieRegressor with [`insurance-cv`](https://github.com/burning-cost/insurance-cv) for correct temporal splits | GUI-driven GLM with built-in one-way/two-way analysis and factor sign-off workflow |
| **GBM modelling** | CatBoost, LightGBM, XGBoost with [`shap-relativities`](https://github.com/burning-cost/shap-relativities) for factor table output | Varies — Akur8 and DataRobot offer GBM with GUI review; Emblem GBM support has historically been limited |
| **Interpretable deep learning** | [`insurance-anam`](https://github.com/burning-cost/insurance-anam) — actuarial neural additive model with per-feature shape functions | Limited native support; typically requires custom integration |
| **Cross-validation** | [`insurance-cv`](https://github.com/burning-cost/insurance-cv) — walk-forward splits with configurable IBNR buffers, sklearn-compatible scorers | Some platforms implement temporal splits; IBNR handling varies |
| **Prediction intervals** | [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — distribution-free finite-sample coverage guarantees; [`insurance-quantile`](https://github.com/burning-cost/insurance-quantile) — quantile and expectile GBMs | Typically point predictions; interval estimation not standard |
| **Rate optimisation** | [`rate-optimiser`](https://github.com/burning-cost/rate-optimiser) — LP efficient frontier with movement caps and GIPP constraints; [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) — SLSQP for large factor spaces | Radar/Earnix specifically targets rate optimisation with integrated demand modelling; Emblem has optimisation add-ons |
| **Demand and price elasticity** | [`insurance-demand`](https://github.com/burning-cost/insurance-demand) — conversion and retention modelling; [`insurance-elasticity`](https://github.com/burning-cost/insurance-elasticity) — causal DML elasticity estimation | Typically available as part of commercial optimisation modules; methodology often opaque |
| **Model validation** | [`insurance-validation`](https://github.com/burning-cost/insurance-validation) — structured PRA SS1/23 reports covering nine sections, HTML and JSON output | Validation reporting features vary; PRA SS1/23 alignment is not typically an explicit feature |
| **Model monitoring** | [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) — exposure-weighted PSI/CSI, A/E ratios, Gini drift z-tests with scheduled alerts | Monitoring dashboards are common in enterprise platforms; insurance-specific metrics vary |
| **Causal inference** | [`insurance-causal`](https://github.com/burning-cost/insurance-causal) — double machine learning for deconfounding; [`insurance-elasticity`](https://github.com/burning-cost/insurance-elasticity) — CausalForestDML price elasticity | Not typically offered natively; DataRobot has some causal tooling |
| **Spatial rating** | [`insurance-spatial`](https://github.com/burning-cost/insurance-spatial) — BYM2 postcode-level models borrowing strength from neighbouring areas | GIS and spatial smoothing tools exist in some platforms; BYM2 specifically is uncommon |
| **Fairness / discrimination** | [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) — proxy discrimination auditing mapped to FCA Consumer Duty requirements | Fairness tooling is an emerging area; FCA-specific mapping is generally not standard |
| **Model governance** | [`insurance-mrm`](https://github.com/burning-cost/insurance-mrm) — ModelCard, ModelInventory, GovernanceReport for PRA SS1/23 compliance | Enterprise platforms typically include governance workflows; PRA SS1/23 alignment varies |
| **Deployment** | [`insurance-deploy`](https://github.com/burning-cost/insurance-deploy) — champion/challenger with shadow mode, rollback, and ICOBS 6B.2 audit trail | Enterprise deployment and A/B testing frameworks are standard in larger platforms |
| **Thin-data segments** | [`credibility`](https://github.com/burning-cost/credibility) — Buhlmann-Straub; [`bayesian-pricing`](https://github.com/burning-cost/bayesian-pricing) — hierarchical Bayes with PyMC 5 | Credibility weighting is standard in Emblem; Bayesian methods less common |
| **Interaction detection** | [`insurance-interactions`](https://github.com/burning-cost/insurance-interactions) — CANN, NID, and SHAP-based interaction tests | Two-way analysis standard in GLM platforms; automated detection less common |
| **Synthetic data** | [`insurance-synthetic`](https://github.com/burning-cost/insurance-synthetic) — vine copula portfolio generation; [`insurance-datasets`](https://github.com/burning-cost/insurance-datasets) — UK motor synthetic data | Not typically included; usually requires separate data management tooling |
| **Licence cost** | Free. MIT licence. No usage caps. | Typically five- to six-figure annual licence costs; pricing is negotiated per contract |
| **Support** | GitHub issues, documentation, community. No SLA. | Commercial SLAs, dedicated support, implementation consultancy |
| **UK regulatory specifics** | FCA GIPP (PS21/5), FCA Consumer Duty, PRA SS1/23, ICOBS 6B.2 constraints built into relevant libraries | UK regulatory features vary by vendor; verify specifics with each vendor |

---

## What we do not cover

Being honest about gaps matters more than padding the feature list.

**Rating engine integration.** Commercial platforms often integrate directly with rating engines (Majesco, Guidewire, Duck Creek). Our libraries produce outputs in standard formats (pandas DataFrames, CSV, JSON) but do not have native connectors to these systems.

**GUI.** If your pricing team needs actuaries to interact with models without writing Python, Burning Cost is not the right choice. We are a code-first toolkit.

**Reserving.** We do not cover claims reserving. For open-source reserving tools, look at [chainladder-python](https://github.com/casact/chainladder-python).

**Enterprise support.** We do not offer SLAs, implementation consulting, or guaranteed response times. GitHub issues are monitored but there is no commercial support contract.

**Data infrastructure.** Burning Cost assumes you already have your data in a usable form. We do not provide ETL, data cataloguing, or data quality tooling.

---

## Who this is for

Burning Cost makes sense if:

- Your team is already working in Python or Databricks, or is moving in that direction
- You want to understand and audit your methodology, not take it on trust
- You are a smaller insurer or MGA where a five-figure annual licence is material
- You need UK-specific regulatory outputs (PRA SS1/23, FCA Consumer Duty, GIPP constraints) and want to see exactly how they are implemented
- You want to combine individual libraries with your own workflow rather than adopting a full platform

It probably is not the right choice if:

- Your actuaries are not comfortable with Python and you need a GUI-driven workflow
- You need a vendor support contract for model governance sign-off
- Your rating engine requires a specific integration that a commercial platform already provides
- You need features from a specific commercial platform's ecosystem (Emblem model libraries, Radar optimisation modules, etc.)

---

## Getting started

All 30 libraries are on PyPI. Install any of them individually:

```
pip install shap-relativities
pip install insurance-cv
pip install rate-optimiser
```

The [full library index](/tools/) lists every library with pip install commands, links to GitHub repos, and links to relevant blog posts. Each library ships with a Databricks notebook demo on synthetic UK motor data.

If you are moving from Emblem to Python, the [training course](/course/) covers the transition — GLMs in Python, GBM pricing, SHAP relativities, conformal prediction intervals, and constrained rate optimisation. Twelve modules, free and open, written for pricing actuaries who already know what they are doing.

---

## Frequently asked questions

**Is there a free alternative to Emblem for insurance pricing?**

Yes, with caveats. Burning Cost covers the statistical modelling workflow that Emblem handles: GLMs, GBMs, factor tables, cross-validation, interaction detection, and credibility weighting. What you lose is Emblem's GUI, its tight integration with WTW consulting workflows, and its track record with UK insurers' model governance teams. If your team works in Python and your model governance process is comfortable with code-based evidence, the open-source route is viable.

**Is there a free alternative to Radar for rate optimisation?**

[`rate-optimiser`](https://github.com/burning-cost/rate-optimiser) and [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) cover constrained rate change optimisation — efficient frontier between loss ratio targets and movement caps, with GIPP constraints. What Radar/Earnix specifically offers around demand integration and rating engine connectivity is harder to replicate without custom work.

**Can Python replace Akur8?**

For the modelling part, yes. [`shap-relativities`](https://github.com/burning-cost/shap-relativities) gives you GBM factor tables in GLM format. [`insurance-anam`](https://github.com/burning-cost/insurance-anam) gives you interpretable shape functions per rating factor. The difference is that Akur8 provides a GUI where non-technical actuaries can interact with the model outputs without writing code. If your team can work in Jupyter or Databricks notebooks, Python is a reasonable substitute. If you need the GUI, it is not.

**What about DataRobot?**

DataRobot is a general AutoML platform, not insurance-specific. For pure model-building capability it competes in a broad ML sense. It lacks the actuarial-specific tooling: walk-forward CV with IBNR, Tweedie/Gamma objectives correctly specified for claims data, factor tables in actuarial format, PRA SS1/23 validation reports, or FCA-specific fairness auditing. Burning Cost fills those gaps. DataRobot's strength is its MLOps infrastructure, which is more mature than what we offer.

---

All libraries are at [github.com/burning-cost](https://github.com/burning-cost). Questions or corrections: [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com).

