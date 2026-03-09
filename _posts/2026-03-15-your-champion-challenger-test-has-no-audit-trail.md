---
layout: post
title: "Champion/Challenger Testing with ICOBS 6B.2.51R Compliance"
date: 2026-03-15
categories: [pricing, libraries, regulation]
tags: [champion-challenger, enbp, icobs-6b, fca-ps21-5, shadow-mode, python, insurance-deploy, model-governance, bootstrap, power-analysis]
description: "83% of UK insurers failed FCA record-keeping requirements for ENBP in 2023. Running a champion/challenger test without an audit trail is the same problem wearing different clothes. insurance-deploy is our 25th open-source library: SHA-256 routing, SQLite quote logging, bootstrap LR tests, power analysis, and an ICOBS 6B.2.51R report that an SMF holder can actually sign."
---

You have built a better pricing model. You want to test it against the current one in production. You run quotes through both, compare the outputs, and eventually decide the new model is better. Then you promote it.

What you probably do not have: a record of which model priced which renewal quote, whether those renewal quotes complied with ENBP, a power calculation showing whether you ran the experiment long enough, a statistical test on the loss ratio difference, or a document an SMF holder can sign for the annual attestation.

That is not a champion/challenger test. It is a comparison exercise. They are not the same thing.

[`insurance-deploy`](https://github.com/burning-cost/insurance-deploy) is our 25th open-source library. It provides the infrastructure that a genuine champion/challenger framework requires: deterministic routing, append-only audit logging, ENBP compliance tracking, insurance-specific KPI computation, bootstrap statistical tests, and an ICOBS 6B.2.51R report. It is on PyPI at v0.1.1, 146 tests passing.

```bash
uv add insurance-deploy
```

---

## Why champion/challenger is hard in insurance

In most domains where A/B testing is routine, the reward signal arrives quickly. An e-commerce test produces a conversion rate within hours. A recommendation algorithm gets click feedback within minutes. You accumulate evidence, run a z-test, make a decision.

Insurance pricing has a 12–36 month reward tail. The thing you actually care about (whether the new model correctly prices risk) is not observable until claims develop. Claim frequency gives you a signal at six to nine months if you have enough volume. Developed loss ratio takes twelve months minimum, and that is before you account for IBNR on long-tail lines. If you are comparing models on motor after three months, you are comparing noise.

The power calculation makes this concrete. Assume a mid-size UK motor book: 3,000 bound policies per month to champion, with a 10% challenger split giving 300 per month to challenger. You want to detect a 3 percentage point difference in loss ratio with 80% power at alpha = 0.05. Given typical UK motor LR volatility (sigma approximately 0.26), the challenger arm needs roughly 5,000 bound policies. At 300 per month, that takes 17 months to bind. Add 12 months of claims development. You are looking at 29 months from experiment start to a credible LR comparison.

This is not a caveat buried in a footnote. It is the central operational fact of insurance champion/challenger, and most teams do not know it when they start the experiment.

The library will tell you upfront:

```python
from insurance_deploy import KPITracker, QuoteLogger

logger = QuoteLogger("./quotes.db")
tracker = KPITracker(logger)

pa = tracker.power_analysis("v3_vs_v2", target_delta_lr=0.03)
print(pa["lr_total_months_with_development"])
# 29.4

print(pa["hr_months_to_significance"])
# 5.1

print(pa["notes"])
# ['LR estimate assumes motor sigma_LR ≈ 0.26 and 12-month development period.',
#  'These are point estimates. Run bootstrap_lr_test() once data matures.']
```

Hit rate significance at 5 months. Loss ratio credibility at 29 months. If your experiment is shorter than that, you are making a promotion decision on insufficient evidence. That is worth knowing before you promote.

---

## Shadow mode is the right default

The library has two modes: `shadow` and `live`. Shadow is the default and we think it should stay the default for most UK pricing teams.

In shadow mode, champion prices every live quote. Challenger runs in parallel on identical inputs, and its outputs are logged but never returned to the customer. There is no pricing difference. There is no regulatory risk. You accumulate challenger predictions alongside champion quotes, then compare them statistically once sufficient data has developed.

Live mode routes a configurable fraction of policies (10% by default) to the challenger model, which prices their quotes for real. The library raises a warning when you enable it:

```python
from insurance_deploy import Experiment

exp = Experiment(
    name="v3_vs_v2",
    champion=registry.get("motor", "2.0"),
    challenger=registry.get("motor", "3.0"),
    challenger_pct=0.10,
    mode="live",  # raises FCA Consumer Duty warning
)
# UserWarning: Live mode routes real quotes to challenger model. This may raise
# FCA Consumer Duty (PRIN 2A) fair value concerns — two customers of identical
# risk profile priced differently simultaneously. Obtain legal sign-off before
# enabling live mode in production. Shadow mode (default) carries zero regulatory risk.
```

The Consumer Duty concern is real. PRIN 2A requires firms to ensure customers receive fair value. Two customers with identical risk profiles receiving different prices simultaneously — because one was randomly assigned to a challenger model — is not a clear-cut position. The FCA has not explicitly prohibited A/B pricing tests, but the asymmetry matters: shadow mode has zero regulatory risk; live mode has non-trivial regulatory risk. Get legal sign-off before enabling live in production.

Shadow mode has its own limitation: you cannot observe conversion rate or customer behaviour from challenger pricing, because challenger prices were never shown to customers. Shadow mode answers the question of whether the challenger model prices risk better. For commercial outcomes, you need live mode with all its caveats.

For the vast majority of pricing actuaries we work with, "does the challenger model price risk better?" is the right first question.

---

## Setting up an experiment

The full stack is five classes:

```python
from insurance_deploy import (
    ModelRegistry,
    Experiment,
    QuoteLogger,
    KPITracker,
    ModelComparison,
    ENBPAuditReport,
)

# Register model versions — append-only, SHA-256 verified
registry = ModelRegistry("./registry")

mv_champion = registry.register(
    current_model,
    name="motor",
    version="2.0",
    metadata={"training_date": "2023-06-01", "features": ["age", "ncd", "vehicle"]},
)

mv_challenger = registry.register(
    new_model,
    name="motor",
    version="3.0",
    metadata={"training_date": "2024-01-01", "features": ["age", "ncd", "vehicle", "telematics_score"]},
)

# Set up experiment — shadow mode, 10% split
exp = Experiment(
    name="v3_vs_v2",
    champion=mv_champion,
    challenger=mv_challenger,
    challenger_pct=0.10,
    mode="shadow",
)
```

`ModelRegistry` is append-only by design: you cannot delete or overwrite a registered version. The model file is SHA-256 hashed at registration and verified at load time. This is not pedantry; it is what "we used model version 2.0 to price this renewal" actually means in an audit context. If the file changed between registration and the audit, verification fails.

---

## Routing and why it must be deterministic

The routing mechanism is `SHA-256(policy_id + experiment_name)`, last 8 hex characters converted to an integer, modulo 100. If the result is below `challenger_pct × 100`, route to challenger:

```python
arm = exp.route("POL-12345")
# Always 'champion' or 'challenger' for this policy_id and experiment name
```

This is deterministic: given the same policy_id and experiment name, you get the same routing decision every time, which is the property that makes the audit trail meaningful.

Random assignment (`random.random() < 0.1`) is not reproducible. You cannot reconstruct which model priced a specific renewal quote six months later. For ENBP compliance, this matters: the model that priced the renewal is part of the record you must keep. Hash-based routing is recomputable from first principles at any time.

Assignment is also by policy, not by quote. A policy that routes to challenger on its first quote will always route to challenger within that experiment. A returning customer does not flip between models mid-experiment, which is required for ENBP audit integrity: the pricing model should be consistent across the lifecycle of each policy.

---

## Logging quotes, binds, and claims

Every quote should be logged:

```python
logger = QuoteLogger("./quotes.db")

# On each quote
arm = exp.route(policy_id)
live_model = exp.live_model(policy_id)  # champion in shadow mode

quoted_price = live_model.predict(risk_features)
enbp = calculate_enbp(risk_features)  # your calculation, not the library's

logger.log_quote(
    policy_id=policy_id,
    experiment_name=exp.name,
    arm=arm,
    model_version=live_model.version_id,
    quoted_price=quoted_price,
    enbp=enbp,          # None for new business
    renewal_flag=True,  # True for renewal quotes
    exposure=1.0,
)
```

The ENBP field is worth explaining. The library records the ENBP value you provide. It does not calculate it. ICOBS 6B calculation is your pricing team's responsibility: the correct ENBP depends on your specific rating factors, model configuration, and the FCA's Q&A guidance. What the library does is store the value you provide, flag automatically whether `quoted_price <= enbp`, and surface breaches in the audit report.

When a renewal quote is logged with ENBP provided, the library sets `enbp_flag = 1` if compliant and `enbp_flag = 0` if the quoted price exceeds ENBP. A breach triggers a warning:

```
UserWarning: ENBP breach: quoted_price 445.00 > enbp 438.00 for policy_id='POL-789'.
This will appear as non-compliant in the ENBP audit report.
```

You see it at log time. It is in the database. It appears in the compliance report. It does not disappear.

Bind and claim events are logged separately as they occur:

```python
# When a customer purchases
logger.log_bind("POL-12345", bound_price=425.00)

# When a claim is reported and at each development update
from datetime import date
logger.log_claim(
    policy_id="POL-12345",
    claim_date=date(2024, 8, 15),
    claim_amount=1_850.00,
    development_month=3,    # 3-month development
)
# 12 months later, update with developed figure
logger.log_claim(
    policy_id="POL-12345",
    claim_date=date(2024, 8, 15),
    claim_amount=2_100.00,
    development_month=12,
)
```

---

## KPI tracking by cohort

Once quotes and binds are logged, KPIs are computed by experiment arm:

```python
tracker = KPITracker(logger)

# Tier 1: immediately available
vol = tracker.quote_volume("v3_vs_v2")
# {'champion': {'n': 9124, 'mean_price': 432.10, ...},
#  'challenger': {'n': 1003, 'mean_price': 427.85, ...}}

# Tier 2: at bind
hr = tracker.hit_rate("v3_vs_v2")
# {'champion': {'quoted': 9124, 'bound': 2919, 'hit_rate': 0.320},
#  'challenger': {'quoted': 1003, 'bound': 329, 'hit_rate': 0.328}}

gwp = tracker.gwp("v3_vs_v2")
# {'champion': {'bound_policies': 2919, 'total_gwp': 1261284.00, 'mean_gwp': 432.10},
#  'challenger': {'bound_policies': 329, 'total_gwp': 136766.65, 'mean_gwp': 415.70}}

# Tier 3: 6+ months (warns if immature)
freq = tracker.frequency("v3_vs_v2", development_months=6)

# Tier 4: 12+ months
lr = tracker.loss_ratio("v3_vs_v2", development_months=12)
```

A challenger hit rate of 32.8% versus champion's 32.0% is encouraging. But the challenger mean GWP of £415.70 versus champion's £432.10 is the reason: if the challenger model is pricing risks more accurately and therefore pricing some risks lower, you would expect both higher conversion and lower average premium. Whether the profitability holds up is a question for the developed LR comparison.

The `summary_report()` method returns a DataFrame with all Tier 1 and 2 metrics side by side, formatted for notebook presentation.

---

## The bootstrap LR test

After 12 months of development, run the statistical comparison:

```python
comp = ModelComparison(tracker)

result = comp.bootstrap_lr_test(
    "v3_vs_v2",
    n_bootstrap=10_000,
    development_months=12,
    seed=42,
)

print(result.summary())
```

```
Test: bootstrap_lr_test | Experiment: v3_vs_v2
Champion estimate: 0.6821 (n=2919)
Challenger estimate: 0.6534 (n=329)
Difference (challenger - champion): -0.0287
95% CI: [-0.0541, -0.0033]
p-value: 0.0138

Conclusion: CHALLENGER_BETTER
Recommendation: Challenger shows significantly better loss_ratio (p=0.014,
95% CI [-0.054, -0.003]). Human review recommended before promotion.
Document the promotion decision with reviewer name and date.
```

The bootstrap resamples at policy level, using 10,000 iterations each drawing with replacement from the bound policies in each arm. Policy-level resampling is appropriate because the quantity of interest is the portfolio loss ratio, not the mean of individual policy loss ratios. SPRT-style sequential testing is not appropriate for developed LR: the reward signal is not i.i.d. and the test requires the claims to have developed first.

The confidence interval excluding zero and p-value of 0.014 is evidence that the difference is not random variation. A 2.9 percentage point LR improvement on a book of that size is meaningful.

Three caveats the library surfaces automatically. First: this is a recommendation for human review, not an automatic promotion. Someone with a name and a date needs to make that decision and record it. Second: in live mode, the LR difference may partly reflect adverse selection. If the challenger priced risks differently, the bound cohorts have different underlying risk mixes, and the observed LR difference is not purely attributable to model quality. Shadow mode avoids this entirely. Third: 329 challenger bound policies is on the low side. The bootstrap is valid, but the CI is wide.

The library also provides a two-proportion z-test for hit rate (available earlier, noisier) and a Poisson GLM for claim frequency (available at 6–9 months, a useful early signal):

```python
hr_result = comp.hit_rate_test("v3_vs_v2")
freq_result = comp.frequency_test("v3_vs_v2", development_months=6)
```

---

## The ENBP audit report

This is the part that directly addresses what the FCA found in 2023. In a multi-firm review of 66 UK motor and home insurers, only 11 — 17% — met ICOBS 6B.2.51R record-keeping requirements fully. Twenty-eight firms had records that were insufficiently granular for the SMF holder to confirm compliance. Twenty-seven firms had no evidence that controls were working as intended. The FCA's language was direct: "many smaller firms had few or no records."

The `ENBPAuditReport` generates the document that addresses this:

```python
reporter = ENBPAuditReport(logger)

report = reporter.generate(
    experiment_name="v3_vs_v2",
    period_start="2024-01-01",
    period_end="2024-12-31",
    firm_name="Acorn General Insurance Ltd",
    smf_holder="Jane Smith (SMF4)",
)

print(report)
```

The output is a Markdown document covering: an executive summary table of renewal quote volumes, ENBP coverage, and compliance rate; arm-by-arm breakdown of compliant versus breach quotes; a table of every model version that priced quotes in the period, with quote counts; a policy-level listing of ENBP breaches if any exist; a routing audit section documenting the deterministic methodology; and an attestation section structured for an SMF holder signature.

The routing audit section reads:

```
## 5. Routing Decision Audit

Routing is deterministic: SHA-256(policy_id + experiment_name),
last 8 hex characters modulo 100. Any routing decision can be
recomputed independently from policy_id and experiment_name.
```

That sentence matters. An SMF holder signing the annual attestation can say, with specificity, how routing decisions were made and how any specific routing decision could be reconstructed. That is what the FCA means by "records sufficient to enable the attestation."

The report explicitly references ICOBS 6B.2.51R. It does not claim to calculate ENBP; that caveat is printed in the footer of every report. The library records what you provide; the correctness of your ENBP calculation is upstream.

---

## Integration pattern for Radar shops

For UK personal lines pricing teams running Radar, the natural pattern is a wrapper: Radar handles production pricing as it always has, and the library sits around it as a governance and comparison layer.

```python
import radar_api  # illustrative

def quote_handler(policy_id: str, risk_features: dict, renewal_flag: bool) -> dict:
    # Radar produces champion price
    champion_price = radar_api.quote(risk_features, model_version="champion")
    enbp = radar_api.quote(risk_features, model_version="new_business_equivalent") if renewal_flag else None

    # Challenger is your Python model
    arm = exp.route(policy_id)
    challenger_price = challenger_model.predict(risk_features)

    # Log both — champion prices the live quote
    logger.log_quote(
        policy_id=policy_id,
        experiment_name=exp.name,
        arm=arm,
        model_version=exp.live_model(policy_id).version_id,
        quoted_price=champion_price,  # always champion in shadow mode
        enbp=enbp,
        renewal_flag=renewal_flag,
    )

    return {"price": champion_price, "shadow_challenger": challenger_price}
```

No Radar infrastructure changes. No IT project. The library slots in between the Radar API call and the quote response. The challenger runs in the background. The audit trail is built incrementally.

WTW added Python deployment to Radar in September 2024, which means a Radar Python extension could eventually call the library directly. For now, the wrapper pattern is more practical and does not require the extension.

---

## What the library does not do

The scope boundary is deliberate. `insurance-deploy` handles: model version registration, experiment routing, quote logging, KPI computation, statistical tests, and the ENBP audit report.

It does not handle: model training (use your existing workflow), rate optimisation (see `insurance-optimise`), distributional shift monitoring (see `insurance-monitor`), or causal attribution of rate changes to outcomes (see [`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy)). These are adjacent problems with adjacent libraries. Keeping the boundary clean means the library does one thing reliably rather than five things approximately.

The storage is SQLite by default. SQLite handles 1–10 million rows without difficulty, which covers several years of quote logging for most mid-size UK books. If you are at enterprise scale and hitting limits, the README documents the PostgreSQL adapter pattern: QuoteLogger is designed to be subclassed.

---

## Our view

Champion/challenger testing in insurance is not conceptually hard. The concept is 30 years old. What has been hard is doing it with the infrastructure the UK regulatory environment now requires: a record of which model priced which renewal quote, an ENBP compliance check on each of those quotes, a statistical test robust to the 12-month claims tail, and documentation sufficient for an SMF holder to sign.

No open-source Python library provided all of this before. The commercial platforms that do (DataRobot MLOps, Akur8 Deploy) are either model-specific or infrastructure-heavy (Kubernetes required). Neither is an option for a pricing team running Radar on a desktop or working in Databricks notebooks with sklearn models.

The 29-month timeline to LR significance is real and worth knowing about. Teams that start champion/challenger experiments without this number tend to either promote too early or abandon the experiment before it matures. The power analysis module is not a nice-to-have; we think it is the most important output in the library for setting operational expectations correctly.

---

`insurance-deploy` is open source under the MIT licence at [github.com/burning-cost/insurance-deploy](https://github.com/burning-cost/insurance-deploy). Install with `uv add insurance-deploy`. 146 tests passing. Requires Python 3.10+, NumPy, SciPy, and joblib.
