---
layout: page
title: "insurance-fairness"
description: "Proxy discrimination auditing for UK insurance pricing models. FCA Consumer Duty compliance, Equality Act 2010, and discrimination-free pricing."
permalink: /insurance-fairness/
schema: SoftwareApplication
github_repo: "https://github.com/burning-cost/insurance-fairness"
pypi_package: "insurance-fairness"
---

Proxy discrimination auditing for UK insurance pricing models. Produces documented, evidenced, FCA-mapped analysis that a pricing committee can sign off and that will stand up to an FCA file review.

**[View on GitHub](https://github.com/burning-cost/insurance-fairness)** &middot; **[PyPI](https://pypi.org/project/insurance-fairness/)** &middot; **[Notebook demo](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fairness_audit_demo.py)**

---

## The problem

The FCA's Consumer Duty (PS22/9) requires firms to demonstrate fair value across customer groups. The FCA's multi-firm review of Consumer Duty implementation (2024) found most Fair Value Assessments were "high-level summaries with little substance." Six Consumer Duty investigations are open; two directly involve insurers.

The mechanism creating fair value failures is proxy discrimination. Your postcode rating factor is probably an ethnicity proxy. Citizens Advice (2022) estimated a £280/year ethnicity penalty in UK motor insurance, totalling £213m per year, driven by postcodes that encode protected-characteristic information without anyone explicitly modelling ethnicity.

Every other fairness library is a methodology tool: it corrects model outputs to satisfy a chosen fairness criterion. This one is a compliance audit tool. It produces documented, evidenced, FCA-mapped analysis that will stand up to an FCA file review.

---

## Installation

```bash
pip install insurance-fairness
```

Optional extras:

```bash
pip install "insurance-fairness[pareto]"   # multi-objective Pareto optimisation (pymoo)
```

---

## Quick start

```python
import polars as pl
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
)
report = audit.run()
report.summary()
report.to_markdown("audit_report.md")
```

Proxy detection without a full audit:

```python
from insurance_fairness import detect_proxies

result = detect_proxies(
    df=df,
    protected_col="gender",
    factor_cols=["age_band", "vehicle_group", "occupation"],
)
result.summary()
```

---

## Key API reference

### `FairnessAudit`

The primary entry point. Runs a full audit of a pricing model and returns a structured `FairnessReport`.

```python
FairnessAudit(
    model,              # fitted CatBoost model
    data,               # Polars DataFrame with features and outcomes
    protected_cols,     # list of protected characteristic column names
    prediction_col,     # column name for model predictions
    outcome_col,        # column name for actual outcomes
    exposure_col,       # column name for exposure (car-years)
)
```

- `.run()` — runs the full audit suite; returns `FairnessReport`
- `.report.summary()` — prints a plain-text summary
- `.report.to_markdown(path)` — writes an FCA-mapped Markdown audit report

### Bias metrics

```python
from insurance_fairness import (
    calibration_by_group,       # A/E ratio per group decile
    demographic_parity_ratio,   # log-space parity ratio
    disparate_impact_ratio,     # 80% rule test
    equalised_odds,             # TPR/FPR parity
    gini_by_group,              # Gini coefficient per protected group
    theil_index,                # income inequality metric applied to premiums
)
```

### Proxy detection

```python
from insurance_fairness import (
    detect_proxies,             # full proxy detection suite
    proxy_r2_scores,            # CatBoost R² — catches non-linear proxies
    mutual_information_scores,  # MI between factor and protected characteristic
    shap_proxy_scores,          # links proxy correlation to price impact
    ProxyDetectionResult,       # result container with .summary()
)
```

### Advanced: discrimination-free pricing

```python
from insurance_fairness.optimal_transport import (
    DiscriminationFreePrice,    # Lindholm marginalisation (ASTIN 2022)
    CausalGraph,                # causal path decomposition
    FCAReport,                  # evidence pack for FCA file review
)
```

### Advanced: double fairness (v0.6.0)

Addresses the Consumer Duty Outcome 2 (Price and Value) obligation — action fairness (equal premiums) and outcome fairness (equal loss ratios) are not the same thing and can conflict.

```python
from insurance_fairness import DoubleFairnessAudit

audit = DoubleFairnessAudit(n_alphas=20)
audit.fit(X_train, y_premium, y_loss_ratio, S_gender)
result = audit.audit()
print(result.summary())
fig = audit.plot_pareto()
print(audit.report())   # FCA evidence pack section
```

### `MulticalibrationAudit` (v0.3.7)

Audits and corrects pricing models for multicalibration fairness.

```python
from insurance_fairness import MulticalibrationAudit

audit = MulticalibrationAudit(n_bins=10, alpha=0.05)
report = audit.audit(y_true, y_pred, protected, exposure)
corrected = audit.correct(y_pred, protected, report, exposure)
```

### `MarginalFairnessPremium` (v0.5.0)

Adjusts Expected Shortfall or Wang transform premiums to be marginally fair.

```python
from insurance_fairness import MarginalFairnessPremium

mfp = MarginalFairnessPremium(distortion='es_alpha', alpha=0.75)
mfp.fit(Y_train, D_train, X_train, model=glm, protected_indices=[0])
rho_fair = mfp.transform(Y_test, D_test, X_test)
```

---

## Benchmark

Tested on 50,000 synthetic UK motor policies with a known postcode-ethnicity proxy issue.

| Metric | Result |
|--------|--------|
| Proxy R² (postcode) | 0.78 — catches the proxy |
| Spearman correlation | r = 0.06 — misses it entirely |
| Full audit runtime | 2–10 minutes |
| Proxy R² computation per factor | 15–60 seconds |

The proxy R² approach catches what direct correlation measures miss. A factor can have near-zero Spearman correlation with a protected characteristic while still contributing substantial discriminatory variation to prices.

---

## Related blog posts

- [Your Pricing Model Might Be Discriminating](https://burning-cost.github.io/2026/03/03/your-pricing-model-might-be-discriminating/) — the Lindholm-Richman-Tsanakas-Wüthrich framework, the Citizens Advice data, and what a defensible audit trail looks like
- [FCA Consumer Duty Pricing Fairness in Python](https://burning-cost.github.io/2026/03/20/fca-consumer-duty-pricing-fairness-python/)
- [Fairness Auditing Without Sensitive Attributes](https://burning-cost.github.io/2026/03/20/fairness-auditing-without-sensitive-attributes/)
- [insurance-fairness vs fairlearn](https://burning-cost.github.io/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/)
- [insurance-fairness vs equiPy](https://burning-cost.github.io/2026/03/22/equipy-vs-insurance-fairness/)
- [Discrimination-Free Pricing via Optimal Transport](https://burning-cost.github.io/2026/03/10/insurance-fairness-ot/) — the causal path decomposition and OT correction approach now implemented in `insurance_fairness.optimal_transport`
