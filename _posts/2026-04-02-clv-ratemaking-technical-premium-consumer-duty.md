---
layout: post
title: "CLV-Integrated Ratemaking: When Customer Lifetime Value Belongs in the Technical Premium"
date: 2026-04-02
author: Burning Cost
categories: [pricing, retention]
tags: [clv, ratemaking, retention, consumer-duty, insurance-survival, insurance-optimise, SurvivalCLV, FCA, PS21/11, CAS-2026, lapse-modelling, technical-premium]
description: "Most pricing teams treat acquisition and retention as separate problems. CLV-aware ratemaking integrates expected policy duration, cross-sell probability, and claim cost trajectory into the technical premium. The FCA's fair value framework creates real constraints on where that CLV logic can go."
math: true
---

The standard framing in P&C pricing is that the technical premium is a risk price: expected losses plus expenses plus a capital charge, computed per policy per year. Customer lifetime value is something the distribution team worries about, or possibly a KPI someone tracks in a dashboard that the pricing team has learned to politely ignore. Acquisition is an underwriting problem. Retention is a renewal problem. They are sequential.

This framing is wrong, and the CAS 2026 research programme has published a request for proposals that treats it as wrong. The RFP asks for formal frameworks integrating CLV directly into ratemaking — not as a retention pricing add-on, but as a component of the technical premium itself. We think this is overdue, and we also think it creates a genuine tension with FCA Consumer Duty that most teams have not fully thought through.

---

## What CLV means in an insurance context

CLV in SaaS is relatively clean: known subscription revenue, predictable churn rates, low per-customer claim costs. CLV in insurance is considerably messier.

For a UK motor policy, the CLV over a five-year horizon involves:

- **Expected retention probability per year** — the probability that the customer renews at year $t$, conditional on having renewed up to year $t-1$. This is a survival function, not a static churn rate. NCD level changes over time via a Markov chain. Claim history evolves. Age bands shift. The covariate path matters.
- **Expected claim cost trajectory** — motor claim frequency and severity are not constant over a policy's lifetime. Younger drivers typically improve in risk profile over the first three to five years of driving. For home contents, underinsurance tends to increase as property values drift. The premium-to-cost margin changes over time.
- **Cross-sell probability** — a motor customer who also buys home insurance has a materially different expected lifetime revenue. This is well-understood commercially and routinely ignored in technical pricing.
- **Reinsurance and expense loading** — CLV should be computed on a net-of-reinsurance basis where the treaty structure varies by claim size. A high-frequency, low-severity driver has a different reinsurance-adjusted CLV from a low-frequency, high-severity driver at the same technical premium.

The actuarial CLV formula that accounts for these is:

$$\text{CLV}(x) = \sum_{t=1}^{T} S(t \mid x(t)) \cdot \bigl(P_t - C_t\bigr) \cdot (1+r)^{-t}$$

where $S(t \mid x(t))$ is the survival probability using the projected covariate path $x(t)$, $(P_t - C_t)$ is the premium-minus-loss-cost margin at year $t$, and $r$ is a discount rate. NCD advancement via Markov chain is handled inside the survival model — the `SurvivalCLV` class in `insurance-survival` does exactly this.

This is not SaaS CLV. It is NPV of the expected policy relationship, accounting for the dynamic covariate path, with claim costs netted against premium income at each year.

---

## The building blocks already exist

`insurance-survival` provides `SurvivalCLV`, which wraps a fitted lifelines survival model (Weibull AFT, Cox, or mixture cure) and computes the above formula:

```python
from lifelines import WeibullAFTFitter
from insurance_survival import SurvivalCLV

# aft is fitted on your renewal panel data
aft = WeibullAFTFitter()
aft.fit(renewal_df, duration_col="tenure_years", event_col="lapsed")

clv_model = SurvivalCLV(
    survival_model=aft,
    horizon=5,
    discount_rate=0.05,
)

results = clv_model.predict(
    policies_df,
    premium_col="annual_premium",
    loss_col="expected_loss",
)
# results: policy_id, clv, survival_integral, cure_prob, s_yr1..s_yr5
```

NCD path marginalisation is handled automatically when `ncd_years` is in the policy dataframe — the model advances the NCD state via UK standard 1-step-up / 2-step-back rules and computes $S(t \mid x(t))$ at the projected covariate values for each year. No Monte Carlo required.

The `discount_sensitivity` method answers the specific question that CLV-aware renewal pricing needs: is a £50 loyalty discount justified?

```python
sensitivity = clv_model.discount_sensitivity(
    policies_df,
    discount_amounts=[25, 50, 75, 100],
    price_elasticity=-0.5,  # 10% price cut ≈ 5.4% retention lift
)
# Returns: policy_id, discount_amount, clv_with_discount,
#          clv_without_discount, incremental_clv, discount_justified
```

Where `discount_justified` is `True`, the loyalty discount generates positive incremental CLV — the retention effect outweighs the revenue forgone. This is what Consumer Duty's "fair value" assessment actually wants to see: documented evidence that a pricing decision benefits the customer relationship, not just insurer margin.

`insurance-optimise` provides `RiskInformedRetentionModel` as the companion tool — it adds `loading_ratio` (renewal price / technical price) and `enbp_proximity` (renewal price / ENBP ceiling) as features before fitting the underlying retention model. The interaction between risk price and renewal offer is what the retention model needs to predict lapse probability correctly:

```python
from insurance_optimise.demand import RiskInformedRetentionModel

model = RiskInformedRetentionModel(
    model_type='logistic',
    technical_price_col='technical_premium',
    renewal_price_col='renewal_price',
    enbp_price_col='nb_equivalent_price',
    ncb_col='ncd_years',
)
model.fit(renewal_df)
lapse_probs = model.predict_proba(renewal_df)
```

Feed the fitted `model` into `SurvivalCLV` via `retention_lift_model` in `discount_sensitivity`, and you have a fully integrated CLV-aware discount decision engine. The pieces fit together. The harder question is what to do with them.

---

## Where CLV enters the technical premium

There is a weaker version and a stronger version of CLV-integrated ratemaking.

The weaker version is what the two tools above provide: CLV-aware renewal discounting. You still set the technical premium using standard loss models. The CLV model decides whether to offer a loyalty discount and by how much. The technical premium and the renewal offer are separate. This is operationally clean and FCA-defensible: the risk price is the risk price, and the discount is a commercially justified retention investment.

The stronger version — which the CAS 2026 RFP is explicitly asking about — integrates CLV directly into the new business premium. On this view, the "correct" technical premium for a new customer is not expected loss plus loadings for this policy year. It is a price that achieves a target CLV given your model of how this customer's risk profile and retention probability will evolve. A profitable customer segment at year-one level may be deeply unprofitable on a CLV basis if the segment has high claim frequency trajectory. Conversely, a break-even new business customer who is likely to maintain low claims and high tenure for five years may justify a below-cost-at-acquisition offer.

This is how sophisticated subscription businesses set acquisition prices. It is how some UK motor insurers already think about aggregator pricing — although rarely in a way that is actuarially formalised. The CAS RFP asks for it to be done properly.

The formal integration requires writing the technical premium as:

$$P^* = \argmax_{P} \text{CLV}(P; x, \hat{S}, \hat{C})$$

subject to constraints on regulatory adequacy (the premium must cover expected losses with adequate loading), consumer duty (the premium must represent fair value), and portfolio mix (adverse selection effects at the book level). This optimisation problem is not one-dimensional: $P$ affects $\hat{S}$ (higher premium lowers retention probability) which affects the full CLV path.

`insurance-optimise`'s Pareto frontier optimiser (`pareto.py`) is the right tool for the constrained version. We have not yet wrapped this into a single CLV-premium optimiser — that is the gap the CAS RFP is funding research to fill.

---

## Consumer Duty creates a real constraint

FCA Consumer Duty PRIN 2A.2.14 requires firms to deliver fair value — the price charged must be reasonable relative to the benefit the customer receives. The "benefit" includes the insurance protection itself, the claims service, and any ancillary value. This is product-level assessment, not individual policy assessment, but it is informed by pricing methodology.

CLV-aware ratemaking creates a genuine fair value tension. If you set new business premiums with a CLV target in mind, you are — implicitly or explicitly — deciding that some customer segments subsidise others across time. A segment that is unprofitable at year one but highly profitable at years three through five gets a subsidised entry price. The subsidy comes from segments that are expected to churn before the profitability window opens.

The FCA's fair value guidance does not prohibit cross-subsidy per se. It does require that the pricing rationale demonstrates the customer is receiving value commensurate with price at the time of charging. A customer being asked to pay a premium inflated above technical cost to fund CLV optimisation on another segment is not receiving fair value, regardless of the actuarial elegance of the CLV model.

The honest position is that CLV-optimal pricing and PS21/11 ENBP constraints point in somewhat different directions. PS21/11 limits the renewal premium to the equivalent new business price — this directly caps the retention profitability that the CLV model can realise. The `enbp_proximity` feature in `RiskInformedRetentionModel` surfacing near-1.0 values as a warning is not a minor implementation detail; it is flagging that the operational boundary of the framework is being approached.

We think the correct implementation is:

1. Use CLV analysis to identify segments where the risk-technical premium is delivering negative expected CLV (high early-tenure losses, high churn before profitable years). These are pricing or product design problems, not retention problems.
2. Use CLV analysis to set loyalty discount limits — `discount_justified=True` in `SurvivalCLV.discount_sensitivity` means the discount generates positive incremental CLV. This is Consumer Duty-defensible.
3. Do not use CLV to inflate new business premiums above what the risk model supports. This is the step where Consumer Duty and CLV optimisation genuinely conflict, and we think Consumer Duty wins.

---

## The CAS 2026 research signal

The CAS RFP for CLV-integrated ratemaking is significant not because CAS sets UK regulatory standards — it does not — but because it signals where the actuarial consensus is heading. The same methodology that the CAS will fund research on in 2026 will appear in IFoA working party papers, Lloyd's Market Association guidance, and eventually FCA supervisory expectations over the following two to three years.

Teams that have integrated CLV analysis into their renewal pricing workflow now will be ahead when the methodology is expected rather than optional. Teams that have it integrated into acquisition pricing with Consumer Duty documentation will be ahead further still.

The building blocks exist. `insurance-survival` v0.4.0 has `SurvivalCLV` with NCD path marginalisation. `insurance-optimise` v0.4.5 has `RiskInformedRetentionModel` with ENBP proximity checking. What does not yet exist is a single wrapper that connects CLV-maximising discount allocation to the Pareto frontier optimiser under Consumer Duty constraints. That is the obvious next step, and it is the step the CAS RFP is explicitly asking the industry to formalise.

---

## Install

```bash
pip install insurance-survival  # SurvivalCLV, discount_sensitivity
pip install insurance-optimise  # RiskInformedRetentionModel, demand subpackage
```

The libraries are independent. You can use `SurvivalCLV` with any fitted lifelines survival model — you do not need `insurance-optimise` installed, and vice versa.

Full worked example combining both in the [CLV renewal optimisation notebook](https://github.com/burning-cost/burning-cost-examples/notebooks/clv_renewal_optimisation.ipynb).
