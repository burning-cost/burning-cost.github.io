---
layout: post
title: "Your Champion Model Has Been Running Unchallenged for Three Years"
date: 2026-03-17
author: Burning Cost
description: "Champion/challenger testing is the right way to evaluate pricing model changes. Most teams do it badly or not at all — ad-hoc scripts, no audit trail, no..."
tags: [champion-challenger, model-deployment, pricing, ICOBS, ENBP, fca-compliance, insurance-deploy, audit, model-governance, python, motor, uk-insurance]
---

We recently had a conversation with a pricing actuary at a mid-size UK personal lines carrier. She had been trying to get a new motor model promoted to production for eight months. The challenger had a Gini 3 points higher than the champion in holdout. The holdout was clean — walk-forward, no leakage. The commercial case was obvious. The blocker was not a governance committee. It was the question nobody could answer: "How do we know the challenger actually outperforms in production, and how do we demonstrate that to the SMF holder?"

The answer to that question is champion/challenger testing done properly. And almost nobody does it properly.

---

## What most teams actually do

The typical deployment workflow goes like this. A new model finishes validation. Someone writes a Python script that routes some fraction of incoming quotes to the new model — usually 10%, picked because it sounds reasonable. The routing logic is stored in a config file that nobody documents formally. Quoted premiums from both models are written to a database table, or possibly a spreadsheet, or possibly both, with slight disagreements between them. Six months later, a pricing analyst runs a query, discovers the challenger has a slightly lower loss ratio, and the model is promoted with a sign-off email and a comment in Confluence.

There are at least three things wrong with this:

First, the routing is often not deterministic by policy. A policy quoted on Monday might go to the champion; the same policy requoted on Thursday might go to the challenger, because the routing uses a random number rather than a hash of the policy ID. This destroys the audit trail. You cannot reconstruct which model priced any given renewal.

Second, there is no power analysis. Six months of data at 10% challenger allocation — around 1,800 bound policies at a typical volume — is nowhere near enough to detect a 3-percentage-point difference in loss ratio at the 80% power level with 12-month claims development. The promotion decision is noise. You might have promoted a model that underperforms the champion, or rejected one that outperforms it.

Third, ICOBS 6B.2.51R requires written records demonstrating that renewal prices do not exceed the Equivalent New Business Price for identical risk profiles. The FCA's general insurance pricing attestation review found widespread record-keeping failures across the firms sampled — mostly because their logs were not granular enough to tie a specific premium to a specific model version for a specific policy. An ad-hoc script and a database table is not a compliant audit log.

---

## What shadow mode actually means

Shadow mode is the correct default for champion/challenger testing, and it is not complicated. The champion prices every quote. The challenger runs on identical inputs, its output is logged, but the customer never sees it. There is zero fair value risk under FCA Consumer Duty (PRIN 2A) — you are not simultaneously charging two customers of identical profile differently. There is no adverse selection confound in the comparison data, because the bound cohort is always the champion's bound cohort.

Live mode — where the challenger actually prices its routed fraction of policies — is the alternative. It enables conversion testing: does the challenger's different pricing lead to different hit rates? This is sometimes worth doing. But it introduces adverse selection bias (if the challenger prices differently, the bound cohorts will have different risk profiles) and it does raise Consumer Duty questions. Get legal sign-off before running live mode. The library is opinionated about this and will issue a warning that you have to explicitly suppress.

```python
from insurance_deploy import Experiment

exp = Experiment(
    name="motor_v3_vs_v2",
    champion=champion_mv,
    challenger=challenger_mv,
    challenger_pct=0.10,
    mode="shadow",   # Default. Always start here.
)
```

The challenger percentage determines which policies are routed to the shadow arm. At 10%, one in ten policies has its risk inputs fed to the challenger and the result logged. The routing decision is deterministic.

---

## Deterministic assignment matters more than it sounds

The routing in `insurance-deploy` uses SHA-256(policy_id + experiment_name), takes the last 8 hex characters as an integer, and takes modulo 100. If the result is below `challenger_pct * 100`, the policy is in the challenger arm.

This is stateless. It requires no database of assignments. Any assignment can be recomputed from a policy ID and an experiment name. Assignment is by policy, not by quote — a policy routed to the challenger arm on its first quote is always in the challenger arm for this experiment, whether it quotes once or ten times.

Why this matters: ENBP compliance requires you to demonstrate that a specific model version priced a specific renewal at a specific price. If your routing is random at the quote level, you cannot reconstruct the assignment. A deterministic hash is independently verifiable — an auditor can confirm the assignment from the policy ID without trusting your infrastructure.

```python
# Same policy, always same arm — regardless of when it's called
arm1 = exp.route("POL-12345")   # "challenger"
arm2 = exp.route("POL-12345")   # "challenger" — same answer every time
```

---

## The audit log is not optional

`QuoteLogger` maintains an append-only SQLite audit log. Every quote is written: policy ID, experiment name, arm assignment, model version, quoted price, ENBP value, renewal flag, timestamp.

```python
from insurance_deploy import QuoteLogger

logger = QuoteLogger("./quotes.db")

def handle_quote(policy_id, inputs, renewal_flag=False, enbp=None):
    arm = exp.route(policy_id)
    champion_price = champion_mv.predict([inputs])[0]
    challenger_price = challenger_mv.predict([inputs])[0]  # shadow: logged, not shown

    logger.log_quote(
        policy_id=policy_id,
        experiment_name=exp.name,
        arm=arm,
        model_version=champion_mv.version_id,
        quoted_price=champion_price,
        enbp=enbp,
        renewal_flag=renewal_flag,
    )

    return champion_price  # customer always sees champion price
```

The ENBP compliance check happens at log time: if `quoted_price > enbp` and `renewal_flag` is true, the record is flagged. The library records ENBP; it does not calculate it. Your pricing team calculates the ENBP per ICOBS 6B, passes it in, and the library maintains the record. The separation is intentional — the ENBP calculation is your responsibility, not the logging infrastructure's.

Bind and claim events are logged separately:

```python
logger.log_bind("POL-12345", bound_price=425.0)

# At each claims development stage
logger.log_claim("POL-12345", claim_date=date(2028, 2, 1),
                 claim_amount=1200.0, development_month=3)
logger.log_claim("POL-12345", claim_date=date(2028, 2, 1),
                 claim_amount=1450.0, development_month=12)
```

---

## The ENBP attestation

The FCA's ICOBS 6B.2.51R requires annual attestation by an SMF holder that renewal prices do not systematically exceed the ENBP. `ENBPAuditReport` generates a Markdown report suitable for inclusion in an attestation pack:

```python
from insurance_deploy import ENBPAuditReport

reporter = ENBPAuditReport(logger)
md = reporter.generate(
    "motor_v3_vs_v2",
    period_start="2027-01-01",
    period_end="2027-12-31",
    firm_name="Acme Insurance Ltd",
    smf_holder="Jane Smith",
)
```

The report includes: renewal count, ENBP breach rate, model version distribution across the period, and per-quarter breach tracking. It is not a substitute for your attestation process. It is the documented evidence trail that makes the attestation possible.

---

## The power analysis you should run before starting

This is where most teams get into trouble. They run a champion/challenger test for six months, look at the KPIs, and make a promotion decision. The decision has very low statistical power, and they do not know it.

At 10% challenger allocation with 3,000 bound policies per month:

- The challenger receives roughly 300 bound policies per month
- Hit rate significance (a 2-percentage-point delta, 80% power): approximately 5 months
- Claim frequency significance (a 0.5-percentage-point delta): approximately 10 months
- Developed loss ratio significance (a 3-percentage-point delta with 12-month claims development): approximately 17 months to accumulate data, plus 12 months development — **29 months total from experiment start**

This is not a limitation of the library. It is the physics of insurance data. Loss ratio has a 12-to-36-month reward tail. Any framework claiming to optimise on LR signal faster than this is using a proxy metric — hit rate, frequency, average premium — rather than actual loss ratio. Proxy optimisation in pricing is how you promote a model that looks good on conversion but destroys underwriting margin.

```python
from insurance_deploy import KPITracker

tracker = KPITracker(logger)

pa = tracker.power_analysis("motor_v3_vs_v2", target_delta_lr=0.03)
print(f"Months to LR significance (incl. 12m development): "
      f"{pa['lr_total_months_with_development']:.0f}")
# Months to LR significance (incl. 12m development): 28
```

Run this before the experiment starts. Show your stakeholders the number. If the timeline is unacceptable, the levers are: increase challenger allocation (goes above 20% and Consumer Duty questions intensify), reduce the target delta (means you only detect larger differences), or accept that you are going to promote on hit rate and frequency rather than developed LR. Make that choice explicitly, not by accident.

---

## The statistical test

When you do have enough development:

```python
from insurance_deploy import ModelComparison

comp = ModelComparison(tracker)

result = comp.bootstrap_lr_test("motor_v3_vs_v2", n_bootstrap=10_000, seed=42)
print(result.summary())
# Test: bootstrap_lr_test | Experiment: motor_v3_vs_v2
# Champion estimate: 0.6402 (n=270)
# Challenger estimate: 0.6118 (n=28)
# Difference (challenger - champion): -0.0284
# 95% CI: [-0.0751, 0.0183]
# p-value: 0.2341
#
# Conclusion: INSUFFICIENT_EVIDENCE
```

INSUFFICIENT_EVIDENCE at p=0.234 with n=28 challenger policies is exactly what you would expect from six months of data. The library tells you this honestly rather than producing a p-value that looks significant because the test was run on too little data. If you want the test to return PROMOTE, you need the sample size. There is no shortcut.

---

## The Radar wrapper

Most UK personal lines pricing teams deploy rates via WTW Radar Live. The library is designed to sit as a governance layer around Radar rather than replace it:

```python
import requests
from insurance_deploy import Experiment, QuoteLogger

def get_quote(policy_id, risk_dict, renewal_flag=False, enbp=None):
    # Champion = Radar Live (existing production system)
    radar_response = requests.post(RADAR_LIVE_URL, json=risk_dict)
    champion_price = radar_response.json()["premium"]

    # Challenger = Python model (new model under test)
    arm = exp.route(policy_id)
    challenger_price = challenger_mv.predict([risk_dict])[0]

    logger.log_quote(
        policy_id=policy_id,
        experiment_name=exp.name,
        arm=arm,
        model_version=champion_mv.version_id,
        quoted_price=champion_price,
        enbp=enbp,
        renewal_flag=renewal_flag,
    )

    return champion_price
```

No Radar infrastructure changes. The governance layer sits alongside it.

---

## The model that never gets challenged

There is a failure mode more common than bad champion/challenger testing: no challenger at all.

A model goes live. It validates well. The team moves on to the next project. Two years later, the market has shifted, the book mix has changed, two new rating variables have become available, and the production model has never been reassessed against anything. Nobody re-trains a challenger because the process for running one properly is unclear, the regulatory exposure from a sloppy test is non-trivial, and the original modellers have left.

The cost of this is real but invisible. You are not losing money on a bad model; you are failing to gain margin on a better one. The production model looks adequate because you are not comparing it to anything. The underwriting loss ratio is fine. The renewal retention is within range. There is no smoking gun.

Run a challenger. Log it properly. Know when you have enough data to decide. That is the whole job.

---

`insurance-deploy` is open source under MIT at [github.com/burning-cost/insurance-deploy](https://github.com/burning-cost/insurance-deploy). Requires Python 3.10+. Install with `uv add insurance-deploy`.

---

**Related posts:**
- [Your Champion/Challenger Test Has No Audit Trail](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/) — the original worked example: shadow mode setup, ENBP logging, and why loss ratio significance takes longer than you think
- [Model Validation Is a Checklist, Not a Test](/2027/09/15/model-validation-pra-ss123/) — PRA SS1/23 sign-off and what governance documentation actually needs to contain
- [The Governance Bottleneck for EBM Adoption](/2026/03/16/the-governance-bottleneck-for-ebm-adoption/) — the translation problem between model quality and pricing committee approval
