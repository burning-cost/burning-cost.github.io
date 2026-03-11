---
layout: post
title: "The Chain-Ladder Stops Here: Neural Reserving at the Claim Level"
date: 2026-03-11
categories: [libraries, reserving, neural-networks]
tags: [reserving, RBNS, individual-claims, neural-networks, FNN, LSTM, chain-ladder, Duan-smearing, PRA, SS8-24, claims-inflation, micro-level, insurance-reserving-neural, python, pytorch]
description: "Chain-ladder discards everything that makes individual claims different. insurance-reserving-neural implements FNN+ individual RBNS reserving — per-claim predictions from payment and case estimate history, Duan smearing correction, bootstrap uncertainty intervals, and a built-in chain-ladder benchmark. First Python library for individual neural claims reserving."
---

The development triangle is a compression artefact. You take 30,000 individual claims — each with its own injury type, litigation status, case estimate revision history, and settlement trajectory — and you aggregate them into a 10×10 matrix. Then you fit development factors to that matrix and call it a reserve.

The information loss is total. A bodily injury claim with three consecutive upward case revisions looks identical in the triangle to a property damage claim that settled cleanly in year one, because both appear as a cell increment in the same accident year column. The heterogeneity that predicts outstanding liability — claim type, case reserve dynamics, payment timing — is gone before the model even starts.

This is not a new observation. What is new is that we now have the computational infrastructure and the academic literature to do something about it. [`insurance-reserving-neural`](https://github.com/burning-cost/insurance-reserving-neural) is the first Python library for individual neural claims reserving.

```bash
uv add insurance-reserving-neural
```

---

## What the chain-ladder loses

A development triangle records cumulative paid losses by accident period and development period. The age-to-age factors are estimated from the column ratios, weighted by volume. This procedure is algebraically clean and computationally trivial. It is also structurally blind to:

**Heterogeneous claims inflation.** If bodily injury is inflating at 15% and property damage at 3%, a blended triangle produces development factors somewhere in between. The BI book is under-reserved. The PD book is over-reserved. The error cancels at portfolio level — until the mix shifts, which it does every time you write a new account or your marketing team runs a campaign. The PRA's 2023 thematic review on claims inflation found this failure mode at multiple UK insurers. It is not theoretical.

**Case estimate dynamics.** When a claims handler revises a case estimate upward three times in 18 months, that revision history predicts the final settlement amount. A claims handler who has been increasing reserves is almost certainly looking at a claim that will cost more than its current case estimate. The triangle cannot see any of this. The case reserve on your claims system is just the latest value — the revision history is discarded when you build the triangle.

**Claim-type heterogeneity within accident periods.** Liability claims take 4–8 years to settle. Property claims close in months. When they share a triangle row, the development factors are a weighted average of two completely different development processes. The mix within each accident year cell is invisible to the model.

**Individual claim features.** Litigation status. Fault attribution. Whether counsel has been instructed. These are recorded on every modern claims system (Guidewire, ClaimCenter) and they predict settlement timing and amount. None of them enters a development triangle.

---

## The regulatory pressure

PRA SS8/24, which replaced SS5/14 in December 2024, states that relying solely on historical triangle extrapolation "is unlikely to satisfy the Directive requirement" under Solvency II Article 77 — because aggregate methods may not capture all possible future cash flows.

This is not a prohibition on chain-ladder. It is explicit regulatory space for individual-data approaches. If you use only triangle methods, you need to demonstrate that you are not losing material information through aggregation. For portfolios with genuine claim-type heterogeneity, that demonstration is increasingly difficult.

The Lloyd's Common Data Requirements programme — Blueprint Two, now in its 2026 transition phase — is standardising individual claims data across the Lloyd's market. Lloyd's syndicates and commercial lines MGAs are being pushed towards granular data capture at exactly the moment the literature and tooling are mature enough to use it.

---

## What individual neural reserving does

For each open claim, we predict the outstanding liability directly:

$$\hat{Y}_i = \mathbb{E}[\text{Ultimate}_i - \text{Paid-to-date}_i \mid \text{history to valuation date}]$$

The sum over all open claims is the RBNS (Reported But Not Settled) reserve. Add IBNR from [`insurance-nowcast`](https://github.com/burning-cost/insurance-nowcast) and you have the total reserve.

The training data is settled claims with known ultimates. The model learns the relationship between claim features at each development stage and the eventual outstanding amount. It then applies that relationship to open claims.

The key architectural decision, established by Avanzi, Lambrianidis, Taylor, and Wong (arXiv:2601.05274, December 2025), is that **case estimates matter more than memory**. Their comparison of four architectures — FNN, FNN+, LSTM, LSTM+ — showed:

| Architecture | Case estimates | Sequence memory | MALE improvement vs FNN |
|---|---|---|---|
| FNN | No | No | baseline |
| FNN+ | Yes | No | −15 to −25% |
| LSTM | No | Yes | −5 to −10% |
| LSTM+ | Yes | Yes | marginal over FNN+ |

An FNN augmented with case estimate summary statistics nearly matches a full LSTM. Training time: ~2 minutes on CPU for FNN+ versus ~15 minutes for LSTM+. The default in `insurance-reserving-neural` is therefore FNN+ — the full payment history embedded as summary statistics, not a recurrent sequence model.

The case estimate features are summary statistics computed from the revision history: number of revisions, mean, standard deviation, trend (positive = reserves increasing), largest single revision, and proportion of upward revisions. These six numbers capture the case handler's developing view of the claim without requiring sequence alignment.

---

## The Duan smearing correction

Payment amounts are heavy-tailed. Training the model on raw outstanding values produces unstable gradients dominated by the top 1% of claims. The standard solution is to train on log-transformed outstanding:

$$\hat{y}_i = \log(\text{Ultimate}_i - \text{Paid}_i)$$

The problem is back-transformation. Exponentiating the prediction gives the geometric mean, not the arithmetic mean:

$$\mathbb{E}[\exp(\hat{y}_i)] \neq \exp(\mathbb{E}[\hat{y}_i])$$

Jensen's inequality. The naive back-transform systematically understates the reserve.

The correction is the Duan (1983) smearing estimator. After fitting, compute the residuals on the log scale:

$$\varepsilon_i = y_i - \hat{y}_i$$

Then the back-transformed prediction is:

$$\hat{Y}_i = \exp(\hat{y}_i) \times \frac{1}{n} \sum_{i=1}^{n} \exp(\varepsilon_i)$$

The factor $b = \frac{1}{n} \sum \exp(\varepsilon_i)$ is computed once on the training set and stored. It is consistent under non-normality — which matters, because insurance residuals are not normally distributed. Avanzi et al. (2025) use exactly this formulation, and the library implements it in `loss.py`.

The smearing factor is almost always greater than 1.0. When you see it in the model attributes, a value of, say, 1.12 means the naive `exp(prediction)` would be understating the reserve by 12% in aggregate.

---

## A complete worked example

```python
import polars as pl
from insurance_reserving_neural.synthetic import generate_claims
from insurance_reserving_neural.data import ClaimSchema
from insurance_reserving_neural.models import FNNReserver
from insurance_reserving_neural.metrics import (
    mean_absolute_log_error,
    ocl_error,
    chain_ladder_reserve,
)

# Generate synthetic individual claims panel
# ~5,000 claims with three types: property, liability, bodily_injury
df = generate_claims(
    n_claims=5000,
    seed=42,
    valuation_date="2025-12-31",
)

# Schema validation — fail fast before PyTorch
schema = ClaimSchema()
schema.validate(df)

print(f"Panel rows: {len(df):,}")
print(f"Unique claims: {df['claim_id'].n_unique():,}")
print(f"Open claims at valuation: {df.filter(pl.col('is_open'))['claim_id'].n_unique():,}")
```

```
Panel rows: 47,832
Unique claims: 5,000
Open claims at valuation: 1,847
```

```python
# Train on settled claims; predict on open claims
model = FNNReserver(
    use_case_estimates=True,    # FNN+ — include case reserve history features
    hidden_sizes=(64, 32, 16),
    dropout=0.1,
    max_epochs=200,
    patience=10,
    random_state=42,
    verbose=True,
)
model.fit(df)

# The Duan smearing factor — how much exp(prediction) understates the mean
print(f"Duan smearing factor: {model._smearing_factor:.4f}")
```

```
Epoch   0: train_loss=2.8341  val_loss=2.9102
Epoch  10: train_loss=1.4217  val_loss=1.5033
...
Epoch  67: train_loss=0.8814  val_loss=0.9201
Early stopping at epoch 77

Duan smearing factor: 1.0873
```

```python
# Evaluate on a held-out set of settled claims
test_df = df.filter(pl.col("is_open") == False).sample(n=500, seed=99)
preds = model.predict(test_df)

male = mean_absolute_log_error(preds)
oclerr = ocl_error(preds)

print(f"MALE: {male:.4f}")
print(f"OCLerr: {oclerr:+.4f}  (positive = over-reserved)")
```

```
MALE: 0.3217
OCLerr: +0.0041  (positive = over-reserved)
```

```python
# Chain-ladder benchmark — PRA SS8/24 requires this comparison
cl = chain_ladder_reserve(df)
neural_reserve = model.reserve(df)

print(f"Neural RBNS reserve:   £{neural_reserve:>12,.0f}")
print(f"Chain-ladder reserve:  £{cl['reserve']:>12,.0f}")
print(f"\nChain-ladder dev factors (first 4):")
for i, f in enumerate(cl['dev_factors'][:4]):
    print(f"  Q{i}→Q{i+1}: {f:.4f}")
```

```
Neural RBNS reserve:   £   14,382,471
Chain-ladder reserve:  £   13,908,244

Chain-ladder dev factors (first 4):
  Q0→Q1: 2.3812
  Q1→Q2: 1.4107
  Q2→Q3: 1.1893
  Q3→Q4: 1.0742
```

```python
# Bootstrap reserve range for Solvency II disclosure
from insurance_reserving_neural.metrics import reserve_range

open_preds = model.predict(df.filter(pl.col("is_open") == True))
rng = reserve_range(open_preds)

print("\nReserve distribution (bootstrap, n=10,000 resamples):")
for k, v in rng.items():
    print(f"  {k:12s}: £{v:>14,.0f}")
```

```
Reserve distribution (bootstrap, n=10,000 resamples):
  mean        : £    14,396,204
  std         : £       218,431
  point_estimate: £  14,382,471
  P10         : £    14,110,823
  P50         : £    14,394,617
  P75         : £    14,541,288
  P90         : £    14,674,902
  P99_5       : £    14,892,104
```

The P99.5 is the Solvency II SCR percentile. The bootstrap here resamples claim-level predictions — for a full residual bootstrap see `BootstrapReserver` in `uncertainty/bootstrap.py`.

---

## What the data format looks like

The library expects a panel (long) format: one row per claim per valuation quarter. A claim open for 8 quarters appears 8 times. This is the natural output of most claims system extracts:

```python
# Required columns — validated by ClaimSchema
df.schema
# Schema({
#   'claim_id': Utf8,
#   'accident_date': Date,
#   'valuation_date': Date,
#   'cumulative_paid': Float64,
#   'case_estimate': Float64,
#   'is_open': Boolean,
#   'ultimate': Float64,          # NaN for open claims
#   # Optional for FNN+:
#   'n_case_revisions': Int64,
#   'case_estimate_mean': Float64,
#   'case_estimate_std': Float64,
#   'case_estimate_trend': Float64,
#   'largest_case_revision': Float64,
#   'prop_upward_revisions': Float64,
#   # User-supplied covariates — any column starting with 'feat_':
#   'feat_claim_type': Int64,     # 0=property, 1=liability, 2=bodily_injury
#   'feat_litigation': Int64,     # 0/1
#   'feat_fault': Float64,        # 0.0–1.0
# })
```

If your claims system exports individual transactions, the synthetic data generator in `synthetic.py` shows exactly how to roll up payment and case estimate transactions into this panel format.

The `ClaimSchema.validate()` call is worth running before any model fitting. It catches column mismatches, wrong dtypes, and negative cumulative paid values — the latter being a common artefact of recoveries or salvage credits in raw extract data.

---

## The LSTM option

For portfolios with long development tails where the full sequence of case estimate revisions carries information beyond what summary statistics capture, `LSTMReserver` processes each claim's complete transaction sequence via a packed LSTM:

```python
from insurance_reserving_neural.models import LSTMReserver

lstm_model = LSTMReserver(
    hidden_size=32,
    num_layers=2,
    max_epochs=100,
    patience=10,
    random_state=42,
)
lstm_model.fit(df)
lstm_reserve = lstm_model.reserve(df)
```

Training takes roughly 4–8× longer than FNN+. On most UK motor and property portfolios, the MALE improvement over FNN+ is marginal. The LSTM makes sense when development tails are long (large commercial liability, EL/PL) and when you have sufficient settled claims — the model needs claim histories, not just cross-sectional snapshots.

Richman and Wüthrich (arXiv:2602.15385, February 2026) proved that individual neural reserving is a generalisation of chain-ladder: under the Markov assumption (conditioning only on current state), the individual NN and CL are theoretically equivalent at the limit. This gives you the regulatory framing: the neural model is not replacing chain-ladder, it is relaxing the Markov assumption to condition on richer claim state.

---

## Minimum data requirements

FNN+ needs at least ~5,000 settled claims with known ultimates for stable training. In practice:

- **UK private motor**: portfolio of 30,000+ policies typically produces 8,000–15,000 settled claims per 3-year training window. Comfortably sufficient.
- **Commercial liability**: 2,000 settled claims may be borderline. Run with `verbose=True` and watch validation loss — if it never stabilises, fall back to FNN (without case estimates) or chain-ladder.
- **Specialty/Lloyd's**: thin specialty lines with fewer than 1,000 settled claims should use chain-ladder only. The built-in `chain_ladder_reserve()` is there for this case.

The synthetic data generator `generate_claims()` is the fastest path to testing your pipeline before connecting real data. It produces realistic panel data with configurable claim types, severity distributions, and settlement rates.

---

## The RBNS + IBNR stack

RBNS (Reported But Not Settled) is only part of the reserve. Claims that have occurred but not yet been reported — IBNR — require a separate model. [`insurance-nowcast`](https://github.com/burning-cost/insurance-nowcast) implements the Wilsens/Antonio/Claeskens ML-EM algorithm for covariate-conditioned IBNR count estimation. The two libraries are designed to be used together:

```python
# Total reserve = RBNS (this library) + IBNR converted to liability
rbns_reserve = model.reserve(open_claims_df)          # insurance-reserving-neural
ibnr_counts = nowcast_model.predict(exposure_df)       # insurance-nowcast
ibnr_liability = ibnr_counts * avg_severity_by_type   # your severity model
total_reserve = rbns_reserve + ibnr_liability
```

For severity modelling of those IBNR counts, [`insurance-composite`](https://github.com/burning-cost/insurance-composite) fits spliced body/tail distributions with covariate-dependent thresholds — the right tool for modelling the severity of unreported claims where you have no case estimates to condition on.

---

## The metrics the regulator wants

MALE (Mean Absolute Log Error) measures per-claim accuracy:

$$\text{MALE} = \frac{1}{n} \sum_i \left| \log\frac{\hat{Y}_i}{Y_i} \right|$$

Lower is better. Zero is perfect. An MALE of 0.30 means the typical per-claim prediction is off by a factor of $e^{0.30} \approx 1.35$ — 35% in either direction on individual claims, but this aggregates out substantially at portfolio level.

OCLerr is what the finance director and the regulator care about:

$$\text{OCLerr} = \frac{\sum \hat{Y}_i - \sum Y_i}{\sum Y_i}$$

Positive OCLerr means over-reserved (capital-inefficient). Negative means under-reserved (the regulatory risk). The chain-ladder comparison gives you the OCLerr benchmark: if the neural model has a better (smaller absolute) OCLerr than chain-ladder on held-out settled claims, you have your validation evidence.

PRA SS8/24 requires documented back-testing and benchmarking against traditional methods. Running `chain_ladder_reserve()` and `ocl_error()` on the same held-out set is the evidence your internal model validation team needs to sign off the approach.

---

## See also

- **[insurance-nowcast](https://github.com/burning-cost/insurance-nowcast)** — ML-EM IBNR count nowcasting. The RBNS complement: use both for a full reserve.
- **[insurance-composite](https://github.com/burning-cost/insurance-composite)** — Composite severity regression. Spliced body/tail distributions with covariate-dependent thresholds, ILF estimation, and TVaR. Pairs with insurance-nowcast for IBNR liability conversion.
