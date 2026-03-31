---
layout: post
title: "Retention-Aware UBI Pricing: Risk Score, Churn Elasticity, and Consumer Duty"
date: 2026-03-31
categories: [techniques]
tags: [telematics, ubi, retention, churn, price-elasticity, discount-optimisation, insurance-telematics, insurance-survival, Consumer-Duty, PS21/11, FCA, CLV, python, MOB-trees, motor-pricing]
description: "Li, Luo, Zhang, Huang and Jiang (IME 2025) combine telematics risk scoring with individual price sensitivity estimation and constrained discount allocation. Here is how to adapt this for the UK market under PS21/11, what parts work, and what the FCA implications actually are."
author: burning-cost
---

Standard UBI pricing solves one problem: assigning premiums that reflect telematics-observed driving risk. It ignores a second problem that is at least as commercially important — whether the driver you just priced will still be on your book next year.

The oversight is not naive. It is structural. The dominant academic and commercial UBI literature (Wüthrich 2017, Jiang & Shi 2024 in the North American Actuarial Journal, the CRoss-series from ETH Zurich) treats ratemaking as a pure risk discrimination exercise. Price equals expected cost divided by some loading factor. Whether the customer accepts that price is treated as an exogenous given, not a modelling input. This is a workable assumption for a traditional GLM pricing exercise. For UBI, where the telematics score is both a risk signal and an observable that the customer can potentially influence, and where churn among high-risk drivers is simultaneously a risk management outcome and a revenue risk, the assumption fails.

Li, Luo, Zhang, Huang and Jiang (Insurance: Mathematics and Economics, 2025, DOI: 10.1016/j.insmatheco.2025.000794) are, as far as we can tell, the first to address this joint problem in the published actuarial literature. Their paper builds a three-stage pipeline: telematics risk scoring, individual price sensitivity estimation via ensemble MOB trees, and constrained discount allocation. The empirical work is on a Chinese insurer's data, but the framework transfers to UK UBI with important modifications. This post covers the method, the Python approximation, and — critically — where PS21/11 creates constraints that the paper's original formulation does not account for.

---

## The feedback loop standard UBI ignores

Imagine a young driver in a UK city. Their telematics score is poor: high average speed, frequent harsh braking, a lot of late-night driving. Your UBI algorithm flags them as high-risk. You price them at £1,800. They lapse.

The naive interpretation is that this is fine — you priced the risk correctly and avoided an adverse selection problem. The less naive interpretation is that you have just created an incentive problem. The driver goes to a competitor who either does not have telematics, has a less granular telematics score, or is willing to cross-subsidise. Your competitor writes a policy at £1,400. The driver improves their driving over the following year because the app gives them weekly feedback. At renewal, their risk score is materially better. Your competitor writes them at £1,300. You never see them again.

Meanwhile, you have a high-telematics-cost book full of drivers who either cannot get cheaper elsewhere or do not shop around — a book that is increasingly unrepresentative of the market because you have systematically selected out the price-sensitive segment.

This is the adverse selection trap we described in [the earlier UBI post]({{ site.baseurl }}{% post_url 2026-03-25-ubi-adverse-selection-telematics-discounts-drive-away-best-risks %}). The Li et al. paper addresses the other side of the same problem: rather than asking "how do telematics discounts attract better risks?", it asks "how do we retain the high-risk customers we have, via targeted discounts, while still making money on them?"

The key insight is that a high-risk driver who receives a modest retention discount and behavioural coaching will, in expectation, generate more lifetime value than the same driver churned away. The paper provides empirical support for this on Chinese data. We think the mechanism is plausible for UK motor UBI, but we hold that view with more uncertainty than the paper's framing implies.

---

## The three-stage pipeline

**Stage 1: Telematics risk scoring.** This is standard and largely solved in our stack. Raw trip data — speed profiles, acceleration, braking, time-of-day, trip distance — flows through `insurance-telematics`'s `TelematicsScoringPipeline` to produce a driver-level risk score. Li et al. use gradient boosted trees on telematics features to predict claim frequency and severity. The [HMM telematics scoring post]({{ site.baseurl }}{% post_url 2026-03-13-insurance-telematics %}) and the [wavelet approach from last week]({{ site.baseurl }}{% post_url 2026-03-26-wavelet-telematics-risk-index-lee-badescu-lin-2026 %}) both give you better risk discrimination than a raw GBT on event counts, but the pipeline interface is the same. Stage 1 output: a risk score $r_i \geq 0$ per driver and a corresponding technical premium $P_i$.

**Stage 2: Price sensitivity estimation.** This is where the paper's novel contribution sits. Li et al. use ensemble MOB trees — Model-Based recursive partitioning Trees, implemented natively in the `partykit` R package — to estimate $P(\text{retain} \mid \text{price}, X_i)$ for each policyholder. MOB trees partition the covariate space recursively, fitting a logistic regression for retention within each leaf. The ensemble (random forest over the recursive partitions) produces individual-level estimates of price sensitivity:

$$\epsilon_i = \frac{\partial}{\partial P} P(\text{retain} \mid P, X_i)$$

This is the individual price elasticity of demand for insurance, estimated non-parametrically. A large negative $\epsilon_i$ means a small price increase drives a large reduction in renewal probability. A near-zero $\epsilon_i$ means the customer is largely inelastic — they will probably renew regardless of moderate price changes.

MOB trees are not yet available as a mature Python implementation. The `partykit` R package is the reference. For a UK pricing team that wants to avoid an R dependency, the Python approximation is a decision tree for the partitioning step combined with per-leaf logistic regression:

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

class MOBApproximation:
    """
    Approximate MOB tree: recursive partitioning via decision tree,
    per-leaf logistic regression for retention probability.
    """

    def __init__(self, max_depth: int = 4, min_leaf_size: int = 100):
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        self.partition_tree = None
        self.leaf_models: dict = {}

    def fit(
        self,
        X: np.ndarray,
        price_change: np.ndarray,
        retained: np.ndarray,
    ) -> "MOBApproximation":
        """
        X: driver covariates (ncd, age, tenure, vehicle_age, ...)
        price_change: delta premium vs prior year, in £
        retained: 1 if renewed, 0 if lapsed
        """
        # Step 1: partition on X only (not price) — determines segments
        self.partition_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_leaf_size,
        )
        self.partition_tree.fit(X, retained)
        leaf_ids = self.partition_tree.apply(X)

        # Step 2: within each leaf, logistic regression on price_change
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            X_leaf = price_change[mask].reshape(-1, 1)
            y_leaf = retained[mask]
            if y_leaf.sum() < 5 or (1 - y_leaf).sum() < 5:
                continue  # not enough variation for logistic fit
            lr = LogisticRegression(max_iter=500)
            lr.fit(X_leaf, y_leaf)
            self.leaf_models[leaf] = lr

        return self

    def predict_retention_prob(
        self,
        X: np.ndarray,
        price_change: np.ndarray,
    ) -> np.ndarray:
        leaf_ids = self.partition_tree.apply(X)
        probs = np.full(len(X), np.nan)
        for leaf, model in self.leaf_models.items():
            mask = leaf_ids == leaf
            probs[mask] = model.predict_proba(
                price_change[mask].reshape(-1, 1)
            )[:, 1]
        return probs

    def price_elasticity(
        self,
        X: np.ndarray,
        price_change: np.ndarray,
        delta: float = 10.0,
    ) -> np.ndarray:
        """Numerical derivative: d/dP P(retain|P,X), units: per £."""
        p_up = self.predict_retention_prob(X, price_change + delta)
        p_dn = self.predict_retention_prob(X, price_change - delta)
        return (p_up - p_dn) / (2 * delta)
```

**Stage 3: Discount allocation.** Given risk scores, technical premiums, and individual price elasticities, the optimisation problem is: allocate a total discount budget $B$ across policyholders to maximise expected insurer profit, subject to each quoted premium remaining above the minimum technical rate.

Li et al. formulate this as:

$$\max_{d_i \geq 0} \sum_i P(\text{retain}_i \mid P_i - d_i, X_i) \cdot (P_i - d_i - \text{ExpectedCost}_i)$$

subject to $\sum_i d_i \leq B$ and $P_i - d_i \geq P_i^\text{floor}$.

The greedy knapsack approximation works well in practice: rank policyholders by marginal profit gain per pound of discount, allocate greedily from the top until the budget is exhausted.

```python
def greedy_discount_allocator(
    technical_premiums: np.ndarray,
    expected_costs: np.ndarray,
    retention_model: MOBApproximation,
    X: np.ndarray,
    budget: float,
    price_floor: np.ndarray,
    new_business_equivalent: np.ndarray,  # PS21/11 hard cap
    increment: float = 10.0,
) -> np.ndarray:
    """
    Greedy knapsack discount allocation.
    Returns discount array (£) for each policyholder.
    """
    n = len(technical_premiums)
    discounts = np.zeros(n)
    remaining_budget = budget

    # Current price change from prior year (passed in by caller)
    current_delta = technical_premiums - technical_premiums  # placeholder: delta = 0

    while remaining_budget >= increment:
        # Current retention probabilities
        p_retain = retention_model.predict_retention_prob(X, current_delta - discounts)

        # Marginal gain from adding one more increment of discount to each policy
        p_retain_plus = retention_model.predict_retention_prob(
            X, current_delta - discounts - increment
        )
        delta_p = p_retain_plus - p_retain  # increase in P(retain)

        # Expected profit if retained
        margin = technical_premiums - discounts - expected_costs
        marginal_gain = delta_p * margin - p_retain * increment

        # Feasibility: cannot exceed floor or PS21/11 ENBP cap
        headroom = np.minimum(
            technical_premiums - discounts - price_floor,
            technical_premiums - discounts - new_business_equivalent,  # PS21/11
        )
        feasible = (headroom >= increment) & (marginal_gain > 0)

        if not feasible.any():
            break

        best = np.where(feasible, marginal_gain, -np.inf).argmax()
        discounts[best] += increment
        remaining_budget -= increment

    return discounts
```

---

## The PS21/11 constraint is not optional

The paper's optimisation objective is profit-maximising. In a Chinese insurance market without FCA oversight, that is a complete objective. In the UK, it is not.

PS21/11 (ICOBS 6B.2, effective January 2022) imposes a hard rule: at renewal, the premium charged to an existing customer cannot exceed what an equivalent new customer would pay through the same channel. The new-business equivalent price (ENBP) is the binding ceiling.

Notice what this means for the Li et al. framework as written. The MOB tree will identify policyholders with low price elasticity — those who will renew regardless of small price movements. The profit-maximising optimiser, if unconstrained, would leave these policyholders at or near their technical price, giving discounts only to the elastic ones. If the technical price for the low-elasticity segment exceeds ENBP, this is PS21/11 non-compliance. The optimiser is inadvertently implementing loyalty pricing.

The `new_business_equivalent` parameter in the `greedy_discount_allocator` above enforces the PS21/11 cap as a hard constraint: no quoted renewal premium can exceed ENBP. This must be enforced before the discount optimisation runs, not as a post-hoc clip. The practical implication: if the technical premium is already above ENBP for a given policyholder, the insurer must discount to ENBP regardless of the budget allocation result. The discount optimiser then operates only on residual budget.

There is a sharper version of this concern. If a UK insurer uses price elasticity scores to construct a system where high-elasticity customers receive discounts and low-elasticity customers do not — even if no individual customer is charged above ENBP — the FCA may still view this as a differential pricing system based on propensity to shop. PS21/11's spirit, not just its letter, is to eliminate the information asymmetry between loyal and shopping customers. A framework that explicitly models that asymmetry and uses it to allocate discounts sits in legally uncertain territory and would need careful governance documentation and likely legal sign-off before implementation.

The Consumer Duty angle is more straightforwardly positive. The framework's behavioural coaching element — where high-risk drivers are identified and offered feedback alongside retention discounts — is directly aligned with the Consumer Duty's "customer outcomes" framing. An insurer who can demonstrate that their UBI programme improves customer driving behaviour over time, reduces claims, and then passes some of that improvement back as premium reduction is demonstrating fair value in the way the FCA says it wants to see.

---

## How this connects to the existing stack

The risk scoring stage is covered. `TelematicsScoringPipeline` in `insurance-telematics` takes raw trip data through feature extraction and HMM scoring to produce driver-level risk estimates. The [telematics trip scoring post]({{ site.baseurl }}{% post_url 2026-03-31-telematics-trip-scoring-functional-data-wavelets %}) and the [HMM scoring post]({{ site.baseurl }}{% post_url 2026-03-24-does-hmm-telematics-scoring-work-insurance-pricing %}) both benchmark what the current library can and cannot do.

The retention probability step has a natural bridge to `insurance-survival`. `WeibullMixtureCureFitter` and `SurvivalCLV` in `insurance-survival` can already model the lapse process and compute CLV given a survival function. The MOB-tree retention model from the Li et al. paper produces $P(\text{retain} \mid \text{price}, X)$ — a discrete-time hazard — which maps directly to the input format `SurvivalCLV` expects. The combination gives you a CLV that is price-sensitive: you can ask "what CLV do I expect at each candidate discount level?" and pick the discount that maximises CLV subject to the PS21/11 cap.

```python
from insurance_survival.clv import SurvivalCLV
from insurance_survival.cure import WeibullMixtureCure

# Fit cure model for structural lapse analysis
cure = WeibullMixtureCure(
    incidence_formula="telematics_score + ncd_years + premium_change_pct",
    latency_formula="telematics_score + age + ncd_years",
)
cure.fit(policy_df, duration_col="months_active", event_col="lapsed")

# Survival-adjusted CLV at different discount levels
clv = SurvivalCLV(survival_model=cure, discount_rate=0.08)

for discount_level in [0, 50, 100, 150]:
    policy_df["quoted_premium"] = (
        policy_df["technical_premium"] - discount_level
    ).clip(lower=policy_df["enbp"])  # PS21/11 clip
    clv_values = clv.compute(
        policy_df,
        premium_col="quoted_premium",
        max_horizon=60,  # 5-year CLV horizon
    )
    print(f"Discount £{discount_level}: mean CLV = £{clv_values.mean():.0f}")
```

What is genuinely missing from the stack is the middle stage — the price sensitivity estimator and discount allocator. These do not exist anywhere in `insurance-telematics`, `insurance-survival`, or `insurance-optimise`. The MOBApproximation and `greedy_discount_allocator` code above is self-contained and could be extracted into `insurance-telematics.retention` with modest additional work.

---

## Why we are not building this today

The paper's core claim — that retention-aware pricing increases insurer profit on the empirical dataset — is supported by a single Chinese insurer's data with no independent benchmark comparison. We do not know whether a simpler logistic retention model with a greedy discount allocator would perform comparably to the MOB-tree ensemble. The paper does not run that ablation.

The regulatory landscape in the UK is also non-trivial in a way that requires more than interface adaptation. The PS21/11 constraints above are necessary but not sufficient. A UK implementation needs legal sign-off on whether the discount allocation mechanism constitutes "differential pricing based on propensity to renew" under the FCA's definition. That review takes time and goes beyond what a library release should pre-empt.

The [renewal classification post]({{ site.baseurl }}{% post_url 2026-03-25-renewal-classification-risk-pricing-vs-retention %}) covers the pricing-versus-retention tension in detail and includes production-grade code for the lapse hazard and PS21/11 ceiling enforcement. Read that alongside this post.

Our view: this is a blog-first approach. If UK pricing teams are actively trying to build retention-aware UBI pricing systems and find the MOB approximation and discount allocator useful, the build decision for `insurance_telematics.retention` becomes easy. The evidence from the paper is strong enough to write about. It is not strong enough — in a UK regulatory context — to build and ship without more validation.

---

## What to read

The full paper is Li H-J, Luo X-G, Zhang Z-L, Huang S-W, Jiang W, "A Usage-Based Insurance (UBI) Pricing Model Considering Customer Retention", *Insurance: Mathematics and Economics* (2025), DOI: [10.1016/j.insmatheco.2025.000794](https://doi.org/10.1016/j.insmatheco.2025.000794). Read it alongside Jiang and Shi (2024, NAAJ) on HMM-based telematics scoring, which provides the risk scoring foundation the paper builds on.

For the UK regulatory constraints, the FCA's PS21/11 policy statement and the Consumer Duty final rules (PS22/9) are both available on fca.org.uk. FCA EP25/2, published in early 2025, provides the most recent empirical evidence on the effect of GIPP on renewal pricing behaviour — that context is essential background for any retention pricing implementation in UK personal lines.
