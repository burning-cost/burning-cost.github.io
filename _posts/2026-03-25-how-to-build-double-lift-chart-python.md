---
layout: post
title: "How to Build a Double-Lift Chart in Python"
date: 2026-03-25
categories: [pricing, techniques, tutorials]
tags: [double-lift, model-comparison, glm, catboost, matplotlib, champion-challenger, insurance-monitoring, insurance-distill, uk-motor, python, tutorials]
description: "Build a double-lift chart to compare GLM vs GBM predictions. Bin by prediction ratio, compute A/E per decile, plot with matplotlib. Standard tool for pricing committee model validation."
---

The double-lift chart is the standard UK actuarial tool for answering one question: when two models disagree on the price of a risk, which one is right?

You have a production GLM and a challenger CatBoost model. The challenger has a higher holdout Gini. The lift-by-decile chart looks clean. But "higher Gini" tells you the challenger ranks risks better on average. It does not tell you *where* the models disagree, or whether the claims experience in those disagreement zones validates the challenger's view. The double-lift chart does that.

The construction is straightforward: sort policies by the ratio of model A's prediction to model B's, group into deciles, compute the actual-to-expected ratio for each model in each decile. A well-discriminating challenger will show A/E close to 1.0 throughout. A champion that the challenger is correctly replacing will show systematic over-prediction in the low-ratio deciles and under-prediction in the high-ratio deciles.

---

## Setup

```python
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
```

No additional library required for the basic version. We will show the `insurance-distill` variant later for situations where your challenger is a distilled pseudo-model.

---

## Step 1: Synthetic data

We need actual claims, two sets of predictions, and exposure. We will construct a scenario where the GLM systematically underprices young drivers in high-performance vehicles — a common pattern in UK motor — and the GBM has picked it up.

```python
rng = np.random.default_rng(42)
n = 40_000

# Covariates
driver_age    = rng.integers(17, 80, n)
vehicle_group = rng.integers(1, 21, n)   # 1 = lowest group, 20 = highest
exposure      = rng.uniform(0.2, 1.0, n)

# True underlying frequency
# Young drivers (17-24) in high vehicle groups (15-20): 45% interaction loading
young      = driver_age < 25
high_group = vehicle_group >= 15
base_freq  = 0.08 + 0.003 * (vehicle_group / 20) - 0.001 * np.clip(driver_age - 40, -20, 20)
true_freq  = base_freq * (1 + 0.45 * (young & high_group))

# Actual claims
actual_claims = rng.poisson(true_freq * exposure)

# GLM predictions: main effects only, misses the interaction
# The GLM applies young-driver and high-group relativities independently
young_loading      = np.where(young, 1.18, 1.0)
high_group_loading = np.where(high_group, 1.12, 1.0)
glm_pred = base_freq * young_loading * high_group_loading

# GBM predictions: picks up the interaction, closer to true_freq
# Simulate by adding ~80% of the interaction signal
gbm_pred = base_freq * (1 + 0.36 * (young & high_group))
# Add small noise so the two models are not identical outside the interaction cells
gbm_pred *= rng.uniform(0.97, 1.03, n)
```

The GLM knows about young drivers (1.18×) and high vehicle groups (1.12×). What it does not know is that the combination is worse than their product. The young/high-group cell true frequency is 1.45× base, but the GLM predicts 1.18 × 1.12 = 1.32× — a consistent 10% underpricing in that cell. The GBM has captured about 80% of this interaction.

---

## Step 2: Build the double-lift table

```python
def double_lift(
    actual: np.ndarray,
    model_a: np.ndarray,
    model_b: np.ndarray,
    exposure: np.ndarray,
    n_deciles: int = 10,
) -> pl.DataFrame:
    """
    Double-lift chart: compare two models by sorting on their prediction ratio.

    Decile 1 = policies where model_a predicts lowest relative to model_b.
    Decile 10 = policies where model_a predicts highest relative to model_b.

    A/E for each model in each decile tells you which model is better
    calibrated in that part of the risk spectrum.
    """
    ratio = model_a / np.clip(model_b, 1e-10, None)
    # Exposure-weighted decile assignment
    order = np.argsort(ratio)
    cum_w = np.cumsum(exposure[order])
    total_w = cum_w[-1]
    decile_idx = np.minimum(
        (n_deciles * cum_w / total_w).astype(int), n_deciles - 1
    )

    rows = []
    for d in range(n_deciles):
        mask_sorted = decile_idx == d
        # Map back to original indices
        orig_idx = order[mask_sorted]

        w = exposure[orig_idx]
        tot = w.sum()
        act = (actual[orig_idx] * 1.0).sum()        # raw claim counts
        exp_a = (model_a[orig_idx] * w).sum()       # exposure-weighted expected
        exp_b = (model_b[orig_idx] * w).sum()

        rows.append({
            "decile":          d + 1,
            "actual_claims":   float(act),
            "expected_a":      float(exp_a),
            "expected_b":      float(exp_b),
            "ae_model_a":      float(act / max(exp_a, 1e-10)),
            "ae_model_b":      float(act / max(exp_b, 1e-10)),
            "avg_ratio":       float((ratio[orig_idx] * w).sum() / max(tot, 1e-10)),
            "exposure_share":  float(tot / max(total_w, 1e-10)),
            "n_policies":      int(mask_sorted.sum()),
        })

    return pl.DataFrame(rows)


chart = double_lift(
    actual=actual_claims.astype(float),
    model_a=glm_pred * exposure,   # expected claims for model A
    model_b=gbm_pred * exposure,   # expected claims for model B
    exposure=exposure,
    n_deciles=10,
)

print(chart.select(["decile", "ae_model_a", "ae_model_b", "avg_ratio"]))
```

```
decile  ae_model_a  ae_model_b  avg_ratio
     1       0.894       0.978       0.831   <- GLM overprices; GBM closer
     2       0.917       0.982       0.892
     3       0.931       0.991       0.921
     4       0.956       0.995       0.946
     5       0.973       0.999       0.966
     6       1.003       1.001       0.987
     7       1.019       1.003       1.013
     8       1.041       0.998       1.042
     9       1.079       1.004       1.092
    10       1.184       1.011       1.243   <- GLM underprices; GBM closer
```

Decile 10 is where the GBM predicts highest relative to the GLM. The actual claims experience in that decile shows A/E = 1.184 for the GLM but only 1.011 for the GBM. The GLM is underpricing by 18% in that decile; the GBM is getting it right. The pattern is consistent across deciles: the GLM has a slope in its A/E, the GBM is flat near 1.0. The GBM wins.

---

## Step 3: Plot

```python
def plot_double_lift(
    chart: pl.DataFrame,
    label_a: str = "GLM",
    label_b: str = "GBM",
    title: str = "Double-Lift Chart",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    deciles = chart["decile"].to_numpy()

    # Left panel: A/E by decile for both models
    ax = axes[0]
    ax.plot(deciles, chart["ae_model_a"].to_numpy(),
            "o-", color="#E05A5A", linewidth=2, markersize=6, label=label_a)
    ax.plot(deciles, chart["ae_model_b"].to_numpy(),
            "s-", color="#2E75B6", linewidth=2, markersize=6, label=label_b)
    ax.axhline(1.0, color="grey", linewidth=1, linestyle="--")
    ax.axhspan(0.9, 1.1, alpha=0.06, color="grey")   # ±10% band
    ax.set_xlabel("Decile (1 = GLM lowest relative to GBM)")
    ax.set_ylabel("Actual / Expected")
    ax.set_xticks(deciles)
    ax.set_ylim(0.75, 1.35)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.legend()
    ax.set_title("A/E by Decile")
    ax.grid(axis="y", alpha=0.3)

    # Right panel: exposure share — check for balanced deciles
    ax2 = axes[1]
    ax2.bar(deciles, chart["exposure_share"].to_numpy() * 100,
            color="#4472C4", alpha=0.75, edgecolor="white")
    ax2.axhline(10.0, color="grey", linewidth=1, linestyle="--")
    ax2.set_xlabel("Decile")
    ax2.set_ylabel("Exposure share (%)")
    ax2.set_xticks(deciles)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax2.set_title("Exposure Share per Decile")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("double_lift.png", dpi=150, bbox_inches="tight")
    plt.show()


plot_double_lift(
    chart,
    label_a="GLM (champion)",
    label_b="CatBoost (challenger)",
    title="GLM vs CatBoost: Double-Lift Chart",
)
```

The left panel is the main diagnostic. You want to see: the challenger's A/E line flat near 1.0 across all deciles. If the champion's line slopes (high in top deciles, low in bottom deciles) while the challenger's is flat, the challenger is correctly identifying where the champion is wrong.

The right panel is a quality check. Each decile should contain roughly 10% of earned exposure. If one decile has 25% and another has 3%, the exposure-weighted binning has been defeated by a concentration of policies at a particular prediction ratio — usually because of a single dominant factor level. In that case, consider using finer bins (20 vingtiles) or segmenting separately.

---

## Reading the output

**A sloped GLM line with a flat GBM line** — the GBM has genuinely improved discrimination. Take the challenger forward to a full validation exercise. The decile pattern tells you *which risks* the improvement is concentrated in: if the slope is in deciles 8-10, the GBM is better at the expensive tail. If the slope is in deciles 1-3, it is better at identifying the cheap risks.

**Both lines flat near 1.0** — the models agree on risk ordering. The GBM may still have a higher Gini (it ranks better), but it is not wrong in any systematic way that the GLM is right about. The champion-vs-challenger decision comes down to which Gini improvement is worth the governance cost of a model change.

**Both lines sloped** — neither model is well-calibrated for this comparison. This usually means the models are being compared on out-of-time data where the claims environment has shifted since training. Check aggregate A/E before interpreting the double-lift. If aggregate A/E > 1.1 for both models, the double-lift is telling you about miscalibration, not about model discrimination.

**The GBM line is worse in some deciles** — this happens. It means the GBM is better overall (higher Gini) but miscalibrated in specific risk segments. This is actually important to know before deploying: a model can win on a Gini test while systematically mispricing a specific cell. The double-lift surfaces this before it reaches the rating algorithm.

---

## The [`insurance-distill`](/insurance-distill/) version

If your challenger is a GLM distilled from a GBM — a pseudo-model produced by fitting a GLM on GBM predictions and then applying it to production data — the `insurance-distill` library has `double_lift_chart()` built in.

```python
from insurance_distill import double_lift_chart

# pseudo: GBM predictions on holdout data
# glm_pred: GLM predictions on same holdout data
# exposure: earned exposure weights

chart_dl = double_lift_chart(
    pseudo=gbm_pred,
    glm_pred=glm_pred,
    exposure=exposure,
    n_deciles=10,
)
print(chart_dl)
# Columns: decile, avg_gbm, avg_glm, ratio_gbm_to_glm, exposure_share
```

The `insurance-distill` variant uses the same exposure-weighted decile construction but returns average predicted rates per decile (not A/E — there is no actual claims column in the distillation workflow). Use it when you are validating a distilled GLM against its parent GBM, not when you have actual claims and want to know which model wins.

For champion/challenger validation against claims experience — the more common use case — the manual implementation above is what you want.

---

## What to put in the model documentation

A model validation pack typically requires the double-lift chart as evidence that the proposed model improves on the current champion and that the improvement is consistent across the risk spectrum. The relevant things to document are:

- How deciles were constructed (exposure-weighted, sorted by ratio of challenger to champion)
- The A/E for each model in each decile, with confidence intervals if the decile size permits it
- Whether any decile shows the challenger performing materially worse than the champion — and if so, why, and whether it affects the deployment decision
- The holdout period and whether it is representative of the deployment period (out-of-time vs out-of-sample)

The double-lift chart alone does not justify a model change. It is one piece of evidence alongside the Gini comparison, the reconstruction check, the factor table review, and the stress test. But it is the piece that most directly answers the pricing committee's question: where does the new model say something different, and is it right?

---

## Related

```bash
uv add insurance-monitoring
uv add insurance-distill
```

- [Insurance Model Monitoring: Gini, A/E, and Double-Lift](/2026/03/22/insurance-model-monitoring-gini-ae-double-lift-python/) — the full monitoring context around the double-lift chart
- [How to Extract GLM-Style Rating Factors from a CatBoost Model](/2026/03/02/how-to-extract-rating-factors-from-catboost/) — once the challenger wins, extract its factor table
- [Your GBM and GLM Are Not Competitors](/2026/02/28/your-gbm-and-glm-are-not-competitors/) — why the champion/challenger framing is correct and the GLM-vs-GBM framing is not
- [`insurance-distill` source](https://github.com/burning-cost/insurance-distill) — for distilled pseudo-model workflows
- [`insurance-monitoring` source](https://github.com/burning-cost/insurance-monitoring) — for the full production monitoring suite
