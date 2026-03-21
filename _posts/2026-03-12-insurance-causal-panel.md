---
layout: post
title: "Causal Fixed Effects for Rate Change Evaluation: Using causalfe on Insurance Panel Data"
date: 2026-03-12
categories: [libraries, pricing, causal-inference]
tags: [causalfe, CFFE, panel-data, causal-inference, rate-changes, fixed-effects, DiD, heterogeneous-treatment-effects, pricing, UK, python, polars]
description: "Causal Forests with Fixed Effects for UK insurance panel data. Rate change evaluation by segment - beyond before-and-after loss ratios. causalfe Python."
---

Before-and-after comparisons of loss ratios are the most common form of rate change evaluation in UK personal lines. They are also the most misleading. The treated group was selected for treatment - you raised rates where you thought risk was deteriorating. The macro environment shifted between observation periods. Policyholders who left after the rate increase were not a random draw from the book. Each of these factors biases a naive before-after comparison in a different direction, and they do not cancel out.

Difference-in-differences solves part of this. A proper DiD with unit and time fixed effects controls for baseline differences between treated and untreated schemes, and for common time trends. Used correctly, it gives you a defensible estimate of the average treatment effect. That estimate, assuming parallel trends holds, is causally identified.

The problem is that the average is the part your pricing team already has an intuition about. What they need is the distribution of effects across the book - which segments absorbed the rate increase with no adverse selection, and which ones triggered lapsing among your best risks. The average treatment effect conceals exactly this.

For heterogeneous treatment effects on panel data, there is now a purpose-built Python tool: causalfe.

## What confounds a naive rate change evaluation

Consider a standard situation: you apply a 5% rate increase to motor schemes in the North West in Q3, leaving comparable schemes in the Midlands unchanged. Six months later you compare loss ratios.

The bias sources are layered. Selection into treatment: you raised rates in the North West partly because it was already deteriorating. Treated units were trending worse before treatment, which means the parallel trends assumption - the workhorse of DiD - is under pressure. Compositional change post-treatment: the 5% increase triggered selective lapsing. Young drivers on comparison sites left; older drivers on auto-renewal stayed. The post-treatment loss ratio reflects both the mechanical effect of the rate change and the changed risk mix. Regional macro effects: Q3 in the North West may coincide with adverse weather patterns, claim cost inflation, or a legal funding environment that differs from the Midlands.

Unit fixed effects absorb the time-constant part of unobserved heterogeneity - the fact that North West schemes have always had higher baseline frequency, for instance. Time fixed effects absorb macro shocks that hit all units. What they cannot absorb: unit-specific time trends (if North West schemes were already trending worse than Midlands schemes before treatment), and post-treatment compositional changes in the outcome.

This is the best-case setting for fixed effects panel methods: they handle what can be handled. They are honest about what they cannot.

## What fixed effects give you, and what they do not

The fixed effects panel model is:

```
Y_it = alpha_i + gamma_t + tau * W_it + epsilon_it
```

where `alpha_i` is the unit fixed effect (absorbed scheme-level unobserved risk), `gamma_t` is the time fixed effect (absorbed macro environment), and `tau` is the average treatment effect.

This is DiD generalised to multiple periods. `tau` is identified under parallel trends: in the absence of treatment, treated and control units would have evolved in parallel. It is a strong assumption. It is testable in the pre-treatment period - you can check whether treated and control units were trending together before the rate change - but it cannot be proven for the post-treatment period where it actually matters.

Standard Causal Forests (EconML's `CausalForestDML`, R's `grf`) extend DiD towards heterogeneous treatment effects, estimating a separate tau for each region of the covariate space. But neither was built for panel data. The fixed effects need to be removed at the tree node level, not globally as a pre-processing step. When you pre-residualise and then fit a causal forest, the composition of units within each leaf has changed relative to the root - the node-level fixed effect structure differs from the global structure - and you get spurious heterogeneity driven by residual fixed effects rather than genuine treatment effect variation.

causalfe (Bonhomme, Cai, and Franke, arXiv:2601.10555, January 2026) fixes this with node-level residualisation. At every candidate split in every tree, it re-estimates and removes unit and time fixed effects within that node. This is computationally expensive but structurally correct. The result is a Causal Forest with Fixed Effects (CFFE) estimator that produces honest, locally-valid confidence intervals for heterogeneous treatment effects on panel data.

## Installing and preparing panel data

```bash
uv add causalfe polars
```

causalfe works with NumPy arrays. We use polars for data preparation - it handles panel reshaping cleanly and the lazy evaluation is useful when pulling from a warehouse.

Your data needs to be in long format: one row per (unit, time) observation. For a personal lines evaluation, unit might be scheme ID, product band, or policyholder ID. Time might be quarter or half-year. The minimum structure:

```python
import polars as pl
import numpy as np
from causalfe import CFFEForest

# Load panel in long format
panel = pl.read_parquet("motor_scheme_panel.parquet")

# panel schema:
#   scheme_id: str        - panel unit identifier
#   quarter_id: int       - panel time (1..8)
#   loss_ratio: float     - outcome
#   treated: int          - 1 if scheme received 5% rate increase in Q5
#   avg_age: float        - scheme-average policyholder age at start of period
#   avg_ncb: float        - scheme-average NCB years
#   urban_pct: float      - urban/rural mix
#   renewal_concentration: float  - pct renewing in peak 2-month window
#   prior_loss_ratio: float       - loss ratio in period before treatment
#   policy_count: int     - exposure proxy

# Require minimum panel depth per unit before fitting
MIN_PERIODS = 6
unit_counts = panel.group_by("scheme_id").agg(pl.len().alias("n_periods"))
valid_units = unit_counts.filter(pl.col("n_periods") >= MIN_PERIODS)["scheme_id"]
panel = panel.filter(pl.col("scheme_id").is_in(valid_units))

print(f"Panel: {panel['scheme_id'].n_unique()} schemes, "
      f"{panel['quarter_id'].n_unique()} quarters, "
      f"{len(panel)} observations")
```

The `MIN_PERIODS` filter matters and we will return to it.

## Fitting CFFE

```python
feature_cols = [
    "avg_age", "avg_ncb", "urban_pct",
    "renewal_concentration", "prior_loss_ratio", "policy_count"
]

# Sort for consistent unit/time ordering
panel = panel.sort(["scheme_id", "quarter_id"])

X = panel.select(feature_cols).to_numpy()
Y = panel["loss_ratio"].to_numpy()
W = panel["treated"].to_numpy().astype(float)
units = panel["scheme_id"].to_numpy()
times = panel["quarter_id"].to_numpy()

# Fit - 2000 trees, honest splitting required for valid inference
model = CFFEForest(
    n_estimators=2000,
    min_samples_leaf=5,
    honesty=True,
)
model.fit(X, Y, W, unit_ids=units, time_ids=times)

# Conditional average treatment effects with 90% confidence intervals
tau_hat = model.predict(X)
tau_lower, tau_upper = model.predict_interval(X, alpha=0.10)

panel = panel.with_columns([
    pl.Series("cate", tau_hat),
    pl.Series("cate_lower", tau_lower),
    pl.Series("cate_upper", tau_upper),
])
```

`tau_hat` is the CATE - the estimated causal effect of the rate increase on loss ratio for each (scheme, quarter) observation, given that scheme's characteristics. A CATE of -3.2 means the 5% rate increase reduced loss ratio by 3.2 points for schemes with that covariate profile. A CATE of +4.7 means the rate increase worsened the loss ratio - the selective lapsing of good risks outweighed the mechanical improvement from higher premiums.

`honesty=True` is not optional if you want valid confidence intervals. Honest splitting uses half the training data to grow the tree structure and the other half to estimate leaf-level effects. Without it, the same observations determine both the splits and the estimates, confidence intervals are anti-conservative, and you will draw the wrong conclusions about which segments have precisely estimated effects.

## Interpreting the output for pricing

```python
# Schemes where rate increase demonstrably helped (upper CI below zero)
beneficial = panel.filter(pl.col("cate_upper") < 0.0)

# Schemes where rate increase demonstrably hurt (lower CI above zero)
adverse = panel.filter(pl.col("cate_lower") > 0.0)

print(f"Rate increase clearly beneficial: {beneficial['scheme_id'].n_unique()} schemes")
print(f"Rate increase clearly adverse: {adverse['scheme_id'].n_unique()} schemes")

# What characterises the adverse group?
print("\nAdverse effect schemes - covariate summary:")
print(adverse.select(feature_cols).describe())
```

In practice on motor data, the adverse-effect schemes cluster on older average age, higher NCB years, and high renewal concentration. The story is what you expect: these are desirable risks who are also the most price-sensitive. A 5% rate increase triggers comparison site activity at renewal, the best policyholders within these schemes exit first, and the scheme's average risk quality deteriorates faster than the average premium rises.

Variable importance lets you rank which characteristics drive the heterogeneity:

```python
importance = pl.Series(
    model.feature_importances_,
    dtype=pl.Float64
)
feat_imp = pl.DataFrame({
    "feature": feature_cols,
    "importance": importance,
}).sort("importance", descending=True)

print(feat_imp)
```

If `avg_ncb` and `renewal_concentration` rank first and second, that tells you NCB concentration and renewal timing optionality are the primary drivers of rate elasticity in your book. This is actionable before the next rate action - you can target differential rate increases by renewal-month profile rather than applying flat rate moves across the entire scheme.

## Insurance-specific gotchas

### Short panels

The node-level residualisation removes degrees of freedom at every candidate split. With only three or four periods per unit, you lose too much within-unit variation to estimate fixed effects reliably at the node level. The minimum is roughly six periods; eight or more is comfortable.

For annual personal lines at the policyholder level, three years of data gives three observations per policyholder - marginal at best. Quarterly data at the scheme level is more tractable: two years gives eight quarters. If you are below the minimum, aggregate to a coarser unit (postcode district instead of individual policyholder) or use a longer observation window before fitting CFFE.

### Entry and exit

Schemes or policyholders who enter or exit the panel during the observation window create an unbalanced panel. causalfe handles unbalanced panels - you do not need to restrict to a balanced sample, but you should be deliberate about it.

Units that enter after the treatment period (new schemes opened post-rate-change) should generally be excluded from treatment effect estimation because they have no pre-treatment observation and their unit fixed effect cannot be separately identified from the treatment effect.

Exit is the more operationally important problem: schemes that lapse after the rate increase are precisely the ones where adverse selection is strongest. If you drop them from the analysis, you are conditioning on survival and your CATE estimates are biased toward zero in the adverse segments. Keep exits in the panel for as long as they exist, and caveat your interpretation if exit is substantial.

```python
# Flag units that exit before the panel end
last_period = panel.group_by("scheme_id").agg(
    pl.col("quarter_id").max().alias("last_quarter")
)
panel = panel.join(last_period, on="scheme_id")
panel = panel.with_columns(
    (pl.col("last_quarter") < panel["quarter_id"].max()).alias("exits_early")
)

exiting_schemes = panel.filter(
    pl.col("exits_early") & (pl.col("treated") == 1)
)["scheme_id"].n_unique()
treated_schemes = panel.filter(pl.col("treated") == 1)["scheme_id"].n_unique()

print(f"Treated schemes exiting before panel end: {exiting_schemes} "
      f"of {treated_schemes} ({100 * exiting_schemes / treated_schemes:.1f}%)")
```

If more than 15-20% of treated schemes exit early, the CATE estimates for the treated group reflect the survivor population. This is not a failure of the method - it is a signal about the scale of adverse selection you are dealing with.

### Exposure weighting

causalfe accepts a `sample_weight` argument. For insurance, this should be policy count or earned exposure. A scheme with 500 policies should not carry the same weight as a scheme with 12 when fitting the forest - the latter's loss ratio is far noisier and should receive proportionally less weight:

```python
weights = panel["policy_count"].to_numpy().astype(float)
weights = weights / weights.mean()  # normalise to mean 1.0

model.fit(X, Y, W, unit_ids=units, time_ids=times, sample_weight=weights)
```

Without exposure weighting, small schemes with volatile loss ratios dominate splits on the basis of noise rather than genuine heterogeneity. For a personal lines book at the policyholder level this is less critical (all policies have similar exposure), but at scheme or segment level it is essential.

## CFFE versus SDID: when to use which

Our [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) library implements Synthetic DiD (SDID) for aggregate causal evaluation of rate changes. CFFE and SDID are solving different problems and are better used together than as alternatives.

SDID answers: what was the causal effect of the rate change on the treated segment as a whole? It works at the segment or product level, requires no individual-level panel, and is robust to pre-treatment trend differences because it reweights both units and time periods to construct the best synthetic control. Use SDID when you want a single, well-identified aggregate effect with a clean regulatory narrative - "our North West motor rate increase reduced the loss ratio by 2.8 points (95% CI: 1.1 to 4.5)."

CFFE answers: which kinds of risks responded differently to the rate change? It requires scheme-level or policyholder-level panel data, returns a distribution of treatment effects across the covariate space, and ranks the features that predict elastic versus inelastic response. Use CFFE when you need the segmentation - before the next rate action, to identify which segments can absorb further rate increases and which cannot.

The practical decision rule:

- Fewer than 20 treated units: SDID (CFFE needs sufficient covariate variation across units to find meaningful splits)
- Aggregate regulatory reporting: SDID
- Pre-next-rate-action segmentation analysis: CFFE
- Investigating adverse selection mechanisms post-rate-change: CFFE
- First pass on any evaluation: SDID, then CFFE if the aggregate effect exists and heterogeneity matters

If SDID returns a near-zero average effect, CFFE may still reveal that positive effects in one segment are being cancelled by negative effects in another - which is usually the more important operational finding.

## Limitations

**Parallel trends is untestable post-treatment.** You can check pre-treatment trend alignment between treated and control units - plot period-by-period averages and look for divergence before Q5. If treated and control units were tracking together in the pre-period, the assumption is plausible. But even perfect pre-treatment parallel trends does not prove the assumption holds after treatment. It is a maintained assumption, not a verified fact, and the confidence intervals do not widen to reflect it.

**No time-varying confounders.** Fixed effects remove unit-level unobservables that are constant over time, and time-level shocks that affect all units equally. They do not remove unit-specific time trends. If treatment assignment is correlated with an unobserved trend - you raised rates in the North West precisely because it was already diverging - the CATE estimates are biased. The test is whether treated and control units show parallel pre-trends; if they do not, this entire approach needs revisiting.

**No dynamic treatment effects.** If your rate change was staged - 5% in Q5, another 3% in Q7 - a binary treatment indicator misrepresents the true treatment. Encoding cumulative rate change instead is sensible, but the CATE interpretation changes to the marginal effect of additional rate increase rather than effect of treatment versus control. Staggered designs require care; see Callaway and Sant'Anna (2021) before extending to multi-period treatment roll-outs.

**Short panels in UK insurance.** Personal lines at the policyholder level typically means two to four renewal cycles - barely enough for reliable node-level residualisation. Scheme-level quarterly data over two or more years is the natural fit for CFFE. If your data is inherently short, a standard DiD with carefully chosen controls is often more defensible than CFFE with insufficient within-unit variation.

## Where to start

causalfe is on PyPI and the paper (arXiv:2601.10555) is readable without a heavy econometrics background. The GitHub repository includes a simulation notebook that generates synthetic panel data with known heterogeneous treatment effects - run that before touching real data. It will confirm that CFFE recovers the planted heterogeneity while standard DiD and a naive causal forest on the same data get it wrong when fixed effects are present.

For the insurance workflow: start with scheme-level quarterly data rather than policyholder-level annual data. Confirm your panel has at least six periods per unit before fitting. Apply exposure weighting by policy count. Run SDID from insurance-causal-policy first to establish the aggregate baseline, then run CFFE to characterise the heterogeneity. The combination gives you both the headline number the FCA is asking for under Consumer Duty, and the segmentation that drives the next pricing decision.

The before-after comparison is not wrong. It is just an answer to the wrong question. The question pricing teams need to answer before the next rate action is not "did the rate change work on average?" It is "which segments should we treat differently next time?" That is what CFFE gives you.

---

*causalfe is available via `uv add causalfe`. The underlying paper is Bonhomme, Cai, and Franke (2026), arXiv:2601.10555. For aggregate causal rate change evaluation, see our [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) library (SDID + CS21). Burning Cost has no affiliation with the causalfe authors.*

- [Your Rate Change Didn't Prove Anything](/2026/03/13/your-rate-change-didnt-prove-anything/)
- [DML for Insurance: Benchmarks and When It Beats Naive Regression](/2026/03/09/dml-insurance-benchmarks/)
- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/)

---

## See also

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — DML as the primary causal approach when panel structure is unavailable or insufficient to absorb unobserved confounders
- [Synthetic Difference-in-Differences for Rate Change Evaluation](/2026/03/13/your-rate-change-didnt-prove-anything/) — SDiD as an alternative when you have a clean rate change event and a control group, rather than continuous treatment
- [Continuous Treatment Causal Inference for Insurance Pricing](/2026/03/12/insurance-autodml/) — the `insurance-causal` library; panel fixed effects can be passed as controls alongside the DML nuisance models
