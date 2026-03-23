---
layout: post
title: "Does proxy discrimination testing actually work?"
date: 2026-03-23
author: Burning Cost
categories: [libraries, validation]
description: "We ran the insurance-fairness proxy detection library against a synthetic motor book with planted proxy effects and compared it against the manual correlation check most teams actually use. The gap is larger than we expected."
canonical_url: "https://burning-cost.github.io/2026/03/23/does-proxy-discrimination-testing-work-insurance/"
tags: [fairness, proxy-discrimination, fca, consumer-duty, equality-act, insurance-fairness, validation]
---

The FCA says test for proxy discrimination. Consumer Duty (PRIN 2A) requires firms to monitor whether their products deliver fair value for different groups. The Equality Act 2010, Section 19 independently prohibits indirect discrimination — rating factors that put a protected group at a particular disadvantage without proportionate justification.

So firms test. Most of them run Spearman correlations between their rating factors and whatever proxy they have for protected characteristics. ONS LSOA ethnicity proportions joined to postcode, perhaps. They look at the table, see |r| < 0.25 across the board, tick the box, and move on.

The question is whether that check catches anything. This post gives you a controlled answer: we built a synthetic motor portfolio where we know exactly which factors are acting as proxies and by how much, then ran both the manual correlation approach and the [insurance-fairness](https://github.com/burning-cost/insurance-fairness) library against it. The numbers below are from a reproducible benchmark — you can run it yourself.

---

## The test setup

20,000 synthetic UK motor policies. Six rating factors: `postcode_area`, `vehicle_group`, `ncd_years`, `age_band`, `annual_mileage`, `payment_method`. The protected attribute is a postcode-level diversity score (0–1 continuous), standing in for an ethnicity proxy of the kind Citizens Advice estimated in their 2022 motor insurance analysis.

The proxy is deliberately structured. London postcode areas (E1, N1, SE1, SW9, etc.) are assigned a base diversity score of ~0.70. Outer city postcodes sit at ~0.40. Rural areas at ~0.20. Individual-level noise is added via `Normal(0, 0.08)`, so the relationship is not mechanically perfect — it mirrors the kind of area-level demographic correlation you get from joining ONS Census 2021 data to postcode districts.

The protected attribute is **never given to the pricing model**. Postcode area is a model input because it is a legitimate risk factor (area-based theft and congestion loadings). The question is whether, and by how much, the postcode factor is acting as an ethnicity proxy — carrying protected-characteristic information into prices even though the model does not model ethnicity directly.

This is the Citizens Advice finding in miniature: the insurer is not doing anything obviously wrong, but the postcode loading produces a systematic premium differential that tracks ethnicity at area level.

---

## What the manual check returns

The most common proxy check in UK pricing teams is a pairwise Spearman correlation between each rating factor and the protected attribute. Simple to run, easy to explain to a compliance committee, and — as the benchmark shows — inadequate.

```python
from scipy.stats import spearmanr

factor_cols = ["postcode_area", "vehicle_group", "ncd_years",
               "age_band", "annual_mileage", "payment_method"]

for col in factor_cols:
    arr = encode_for_correlation(df[col])  # integer-encode categoricals
    r, _ = spearmanr(arr, df["diversity_score"])
    print(f"{col:<20} r={r:+.4f}  {'FLAG' if abs(r) > 0.25 else 'OK'}")
```

Result on seed=42, n=20,000:

| Factor | Spearman r | Flag? |
|--------|-----------|-------|
| postcode_area | +0.0634 | OK |
| vehicle_group | +0.0160 | OK |
| ncd_years | -0.0050 | OK |
| age_band | -0.0045 | OK |
| annual_mileage | -0.0034 | OK |
| payment_method | +0.0094 | OK |

**0/6 factors flagged.** The postcode proxy — which we know is there by construction — returns a Spearman r of 0.06. Well below any reasonable threshold.

This is not a cherry-picked seed. We ran 50 independent seeds (each a fresh 20,000-policy draw). The manual Spearman check missed the postcode proxy in all 50. Mean |r| across seeds: 0.063. Maximum |r|: 0.066. The proxy is structurally undetectable by rank correlation because the relationship is non-linear and categorical: postcode area encodes group identity, not a monotone ordering that Spearman is designed to measure.

---

## What the library returns

The library runs three methods in sequence: CatBoost proxy R-squared, mutual information, and SHAP proxy scores.

```python
from insurance_fairness.proxy_detection import (
    proxy_r2_scores,
    mutual_information_scores,
    shap_proxy_scores,
)

r2 = proxy_r2_scores(
    df=df,
    protected_col="diversity_score",
    factor_cols=factor_cols,
    exposure_col="exposure",
    catboost_iterations=80,
    catboost_depth=4,
    is_binary_protected=False,
    random_seed=42,
)

mi = mutual_information_scores(
    df=df,
    protected_col="diversity_score",
    factor_cols=factor_cols,
    is_binary_protected=False,
)
```

For each factor, proxy R-squared fits a CatBoost model predicting the protected attribute from that factor alone. R-squared > 0.10 is amber; > 0.25 is red.

Result on seed=42:

| Factor | Proxy R2 | MI (nats) | Status |
|--------|---------|-----------|--------|
| postcode_area | **0.7767** | **0.8169** | **RED** |
| vehicle_group | 0.0000 | 0.0019 | GREEN |
| ncd_years | 0.0000 | 0.0063 | GREEN |
| age_band | 0.0000 | 0.0025 | GREEN |
| annual_mileage | 0.0000 | 0.0056 | GREEN |
| payment_method | 0.0000 | 0.0038 | GREEN |

**1/6 factors flagged — correctly.** Proxy R-squared of 0.78 for postcode area means a CatBoost model predicting the diversity score from postcode alone explains 78% of the variance. Mutual information of 0.82 nats confirms it independently. The other five factors show R-squared of zero — no false positives.

The Monte Carlo confirms the finding is structural. Library detection rate across 50 seeds: **50/50 (100%)**. Mean proxy R2: 0.769, std 0.008, range [0.752, 0.796].

Total benchmark time: 4.1 seconds for all six factors including SHAP proxy scores.

---

## Why the gap is so large

Spearman measures rank correlation. It is the right tool for asking "does this numeric variable tend to increase as the protected attribute increases?" It is the wrong tool for asking "does this categorical variable encode protected-characteristic information?"

Postcode area is a categorical variable with 50+ levels. The diversity score is not monotone in postcode area ordering — E1 and SW1 both have high diversity scores, while AL1 and LA1 have low ones. There is no consistent rank ordering. Spearman sees noise; it returns ~0.06.

CatBoost, by contrast, can learn an arbitrary mapping from postcode label to diversity score. It discovers that "E1, N1, SE1, SW9" maps to ~0.70 and "AL1, DL1, LA1" maps to ~0.20. A CatBoost model with 80 iterations and depth 4 is more than powerful enough to recover that structure from 20,000 policies. R-squared of 0.78 is the result.

This matters because proxy discrimination in insurance is almost always categorical in structure. The classic cases — postcode encoding ethnicity, vehicle type correlating with age, payment method correlating with income — are not monotone relationships between continuous variables. They are group-identity encodings in categorical factors. Spearman was not designed for this. The library was.

---

## Does detection connect to real money?

Proxy detection is only actionable if it connects to a quantifiable premium differential. The benchmark portfolio, with postcode area loading contributing `base_diversity * 0.25` to the log-premium, produces the following by diversity group:

| Group | N policies | Mean diversity | Mean premium |
|-------|-----------|---------------|-------------|
| Low (<0.33) | 4,138 | 0.220 | £515.63 |
| Mid (0.33–0.60) | 6,434 | 0.449 | £538.65 |
| High (>=0.60) | 9,428 | 0.716 | £577.73 |

High-diversity policyholders (predominantly inner London) pay approximately **14% more** than low-diversity policyholders, with the postcode-area channel contributing roughly £70–90 per policy of that differential — the portion that cannot be defended on risk grounds if postcode is confirmed as an ethnicity proxy.

Scaled to 9,428 high-diversity policies, that is roughly **£500,000–600,000 per year** in potentially indefensible premium loading. The order of magnitude matches Citizens Advice's 2022 estimate (£213m total for UK motor, ~£280 per policy — our benchmark is a proportional slice).

The library reports this figure automatically. `FairnessAudit.run()` produces a Markdown report with the premium differential table, the proxy detection results, and explicit regulatory cross-references (PRIN 2A, Equality Act s.19), formatted for a pricing committee pack.

---

## SHAP proxy scores: the price-impact link

Proxy R-squared tells you whether a factor carries protected-characteristic information. It does not tell you whether that information is propagating into prices. A factor can have high proxy R-squared but low weight in the model — present but not harmful.

The SHAP proxy score completes the chain. For each factor, it computes the Spearman correlation between the factor's SHAP contribution to the model's price prediction and the protected attribute. A high SHAP proxy score means the factor is not just correlated with the protected characteristic — it is actively routing that correlation into the price output.

For postcode area on this benchmark: **SHAP proxy score 0.75**. The premium impact of postcode tracks the diversity score almost perfectly. The proxy relationship is not dormant. It is live in the prices.

The other five factors all return SHAP proxy scores below 0.05 — correctly, since none of them have meaningful proxy R-squared.

---

## Limitations

This is synthetic data. The proxy relationship is planted by construction: postcode area encodes diversity score by design. Real portfolios are messier. The diversity score is continuous and area-level — it does not assign ethnicity to individuals. That matters both statistically (the proxy is diffuse, not deterministic) and legally (the Equality Act protects individuals, not statistical aggregates).

A high proxy R-squared flags a factor for review. It is not proof of indirect discrimination. Demonstrating indirect discrimination under s.19 requires showing that the provision, criterion or practice puts persons sharing a protected characteristic at a particular disadvantage — which requires the full causal chain from factor to outcome, not just a statistical association. The library is a detection tool, not a determination.

On the regulatory interpretation question: Consumer Duty and the Equality Act impose overlapping but distinct obligations. The FCA's TR24/2 (August 2024) found most insurers' Fair Value Assessments were "high-level summaries with little substance" — the evidential bar is not being met. The library's structured Markdown output addresses that gap. But what you do with a flagged factor — whether you remove it, restrict it, or justify it as a proportionate means to a legitimate aim — is a legal and pricing judgement that no library can make for you.

Fit time for the full `FairnessAudit.run()` with proxy detection scales with portfolio size and model complexity. On 20,000 policies with six factors, proxy R-squared with CatBoost runs in 0.5 seconds; the full audit including SHAP proxy scores finishes in under 5 seconds. We have not benchmarked on books above 500,000 policies.

---

## What this series is for

This is the first post in a validation series we are running alongside the Burning Cost libraries. Each post takes one library, builds a dataset where the ground truth is known, and asks whether the library detects what it claims to detect.

The honest answer for insurance-fairness proxy detection: it works, and the comparison against manual correlation analysis is not close. Spearman missed a planted proxy in all 50 seeds. The library caught it in all 50. That gap is large enough to matter in a real compliance workflow.

The library is at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). The benchmark code is in `benchmarks/benchmark.py` and `benchmarks/benchmark_sensitivity.py`.

```bash
pip install insurance-fairness
python benchmarks/benchmark.py
```

Run it. The seed=42 results above are not cherry-picked — the sensitivity script will confirm.
