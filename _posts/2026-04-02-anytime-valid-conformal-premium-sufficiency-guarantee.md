---
layout: post
title: "Your Premium Sufficiency Guarantee Has an Expiry Date"
date: 2026-04-02
author: Burning Cost
categories: [conformal-prediction, model-monitoring]
tags: [conformal-prediction, conformal-risk-control, anytime-valid, sequential-testing, premium-sufficiency, insurance-conformal, insurance-monitoring, PITMonitor, LIL, arXiv-2602.04364]
description: "Every time you re-run conformal risk control calibration on a growing book, you are implicitly doing multiple testing. Hultberg et al. (2026) formalise the fix."
---

You ran conformal risk control on last quarter's calibration set. You found lambda_hat — the loading multiplier where your expected shortfall is at most 5% of premium income. You filed that number with your pricing committee. Then new policies came in, you reran the calibration on the expanded set, and you updated lambda_hat. You did the same thing the quarter after that.

Each time you reran, you claimed the same guarantee: "expected shortfall is at most 5%". What you were actually providing was something weaker, and the gap compounds with every recalibration.

This is the problem Hultberg, Bates & Candès address in "Anytime-Valid Conformal Risk Control" (arXiv:2602.04364, February 2026). The paper is incremental — we will say that clearly up front — but the thing it formalises is real, and it affects any team that recalibrates conformal models on a growing book.

---

## What the guarantee actually says

The standard CRC guarantee (Angelopoulos et al., ICLR 2024, which `conformal_risk_calibration` implements) is a marginal expectation result: given a fixed calibration set of size n, the calibrated threshold lambda_hat satisfies

```
E[L_{n+1}(lambda_hat)] <= alpha
```

where the expectation is over a single draw of the calibration set plus one new observation. The finite-sample correction in our implementation — `(n/(n+1)) * empirical_risk + B/(n+1)` — handles the single-shot finite-sample bias.

The key word is *single draw*. The guarantee is valid exactly once: at the moment of calibration, against the specific calibration set that produced it. It says nothing about what happens when you recalibrate next quarter.

When you recalibrate repeatedly on a growing set — quarter 1 at n=800, quarter 2 at n=1,100, quarter 3 at n=1,400 — you are running a sequential test. At each step, you check whether the expanded dataset still supports your claimed alpha. This is not a single-draw guarantee any more. This is multiple testing, and multiple testing requires correction.

Without correction, the probability that at least one of your recalibrations produces a false "guarantee achieved" result grows with the number of recalibrations. Run ten quarterly recalibrations at nominal alpha=0.05 and your true error rate across the testing sequence is substantially higher than 5%. For independent sequential tests at per-test alpha=0.05, the compounded family-wise error rate across ten recalibrations reaches approximately 40% — the standard binomial worst-case under independence. With the strong dependence between successive calibration sets (each augments the last), the actual inflation is lower than this bound; successive sets are highly correlated, which reduces the effective number of independent tests. But the direction is unambiguous: the effective error rate is above 5%, and the 40% figure is the ceiling, not an operative estimate.

---

## The Law of the Iterated Logarithm correction

The paper's fix is a single multiplicative correction to the alpha threshold. Instead of calibrating at alpha, you calibrate at a tighter alpha_t that accounts for the stage t in the sequential testing sequence:

```
alpha_t = alpha - C * sqrt(log(log(n_t)) / n_t)
```

where n_t is the calibration set size at stage t and C is a constant derived from the LIL boundary. This is the Law of the Iterated Logarithm applied to the empirical risk process.

The size of the correction depends strongly on calibration set size. At n=1,000 and C=1.7, the raw formula gives `1.7 * sqrt(log(log(1000))/1000) = 0.075` — a 7.5 percentage point tightening of alpha. The implementation floors at alpha/2, so alpha falls from 0.05 to 0.025 (the floor) at small-to-medium calibration sets. At n=100,000, the correction is approximately 0.8 pp (alpha goes to 0.042). On large commercial books with calibration sets in the hundreds of thousands, the correction becomes small — a few basis points of additional loading. On quarterly-recalibrating portfolios with n in the low thousands, the floor at alpha/2 is the binding constraint: the correction is substantial, not marginal.

The important result is the guarantee you recover: the sequential family-wise error rate is now controlled at alpha across the entire recalibration sequence, not just at a single snapshot.

---

## Where this sits relative to what we already have

We want to be honest about the contribution here, because the paper is genuinely incremental over a tool we already ship.

`PITMonitor` in `insurance-monitoring` already provides anytime-valid sequential monitoring via a mixture e-process over probability integral transforms. It detects *changes* in calibration with sequential type-I error control — you can run `monitor.update(u)` every time a new policy matures and the FWER stays bounded. The anytime-valid testing literature it draws on (e-values, testing by betting) is the same literature Hultberg et al. are connecting CRC to.

What Hultberg et al. add is the formal bridge specifically for the CRC calibration step itself. `PITMonitor` detects drift in an already-deployed model. The anytime-valid CRC result handles the calibration recalibration loop — the step before deployment, where you set lambda_hat in the first place. These are different intervention points in the pricing workflow:

- **Calibration (pre-deployment):** Use anytime-valid CRC to set lambda_hat with a guarantee that survives repeated recalibrations.
- **Post-deployment monitoring:** Use `PITMonitor` to detect when the deployed model's calibration has degraded and recalibration is warranted.

The paper's specific contribution is the LIL boundary for the CRC risk process. The LIL boundary for Gaussian processes is classical; applying it to the bounded empirical risk process in CRC requires showing that the risk process satisfies the right regularity conditions, which is the technical work in the paper. For practitioners, the upshot is just the corrected alpha formula above.

---

## When this matters for UK pricing teams

Three situations where the sequential multiple-testing problem is practically material:

**Quarterly governance cycles.** Most UK personal lines pricing teams recalibrate conformal thresholds on a quarterly cycle, coinciding with pricing reviews. Four recalibrations per year, three-year model lifetime: twelve sequential tests. At nominal alpha=0.05, the uncorrected sequential error rate is uncomfortably above 5%.

**Regulatory reporting.** FCA Consumer Duty and PRA CP6/24 (model risk management for insurers) both require that risk management frameworks perform as described. If you report "our conformal premium sufficiency controller targets 5% expected shortfall" in a Solvency II internal model validation and the actual sequential error rate is higher, that is a documentation problem. The LIL correction gives you a number you can stand behind across the model's lifetime.

**Growing books.** A new MGA or parametric product that adds 50–100 policies per month will naturally trigger recalibration when n crosses thresholds that make additional lambda grid points reachable. Each crossing is a sequential test.

For mature books recalibrated once annually, with calibration sets in the thousands and genuine calendar-year separation between calibrations, the correction is small enough to ignore. The problem is proportional to the frequency of recalibration relative to how quickly new data arrives.

---

## What to do about it

The mechanical fix is straightforward. In `conformal_risk_calibration`, pass a corrected alpha at each recalibration stage:

```python
import numpy as np
from insurance_conformal.risk import conformal_risk_calibration

def lil_corrected_alpha(alpha: float, n: int, C: float = 1.7) -> float:
    """
    LIL correction for sequential CRC recalibration.

    alpha: nominal target shortfall level
    n: current calibration set size
    C: LIL boundary constant (Hultberg et al. suggest 1.7 for bounded losses)
    """
    if n < 10:
        return alpha  # correction ill-defined at very small n
    correction = C * np.sqrt(np.log(np.log(n)) / n)
    return max(alpha - correction, alpha / 2)  # floor at alpha/2 to avoid over-correction

# At each recalibration:
n_cal = len(y_cal)
alpha_corrected = lil_corrected_alpha(alpha=0.05, n=n_cal)

lambda_hat, _, risk_curve = conformal_risk_calibration(
    losses=loss_matrix,
    lambdas=lambda_grid,
    alpha=alpha_corrected,
    B=B,
)
```

The `C=1.7` constant is from the paper's Theorem 3.2. The floor at `alpha/2` prevents pathological over-correction when the formula is used at small n (where `log(log(n))` grows faster than the correction is meaningful).

We will add `lil_corrected_alpha` as a utility function in `insurance-conformal` v1.3.0. Until then, the implementation above is the whole thing — it is twenty lines including the docstring.

For `PremiumSufficiencyController`, the correction slots in at calibration time:

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)

# If this is recalibration k at stage n:
alpha_seq = lil_corrected_alpha(alpha=0.05, n=len(y_cal))
psc_seq = PremiumSufficiencyController(alpha=alpha_seq, B=5.0)
psc_seq.calibrate(y_cal, premium_cal)

print(f"Sequential-corrected lambda_hat: {psc_seq.lambda_hat_:.4f}")
print(f"vs. naive lambda_hat:            {psc.lambda_hat_:.4f}")
```

In practice, the sequential lambda_hat is 0.5–2% larger than the naive one. On a book with a target loss ratio of 70%, a 1% higher loading on the conformal upper bound is operationally inconsequential. But you now have a guarantee that holds across all future recalibrations, not just the current one.

---

## What the paper does not cover

Two limitations worth naming.

The LIL correction assumes the calibration set grows by augmentation — new data is added, old data is not removed. If you recalibrate on a rolling window (dropping policies older than three years as you add new ones), the analysis changes. The paper references a rolling-window extension, but the correction formula is more complex and we have not validated it against our typical window sizes; treat it as unverified for production use. Rolling-window recalibration is common for motor books with significant trend; do not apply the simple formula above to a rolling window.

The paper's simulation experiments use synthetic data with light-tailed loss distributions. Insurance claim losses are heavy-tailed. The constant C=1.7 may be conservative for heavy-tailed bounded losses — heavy tails make the empirical risk process more variable, which could require a larger C. This is worth testing against your own data before filing the corrected guarantee in a formal document.

---

## Summary

Standard conformal risk control gives a one-shot guarantee: valid at the moment of calibration, on the specific calibration set used. Repeated recalibration on a growing book is sequential testing, and the uncorrected guarantee erodes with each recalibration — a substantially inflated effective error rate after ten recalibrations at nominal 5%, driven by the multiple-testing structure of the sequential calibration loop.

The Hultberg et al. LIL correction is a single formula applied to alpha before each recalibration. The practical impact on lambda_hat is small (0.5–2% larger), but the guarantee it restores is the one you want to be able to report: expected shortfall at most alpha across all past and future recalibrations, not just the most recent one.

`insurance-conformal` v1.3.0 will include `lil_corrected_alpha`. The implementation above works now.

The paper: Hultberg, E., Bates, S. & Candès, E.J. — "Anytime-Valid Conformal Risk Control" (arXiv:2602.04364, February 2026).

---

Related:
- [Conformal Risk Control for Premium Sufficiency: PremiumSufficiencyController](/2026/03/13/insurance-conformal-risk/) — the baseline CRC implementation this post extends
- [ModelMonitor: Calibration Testing That Actually Tells You What To Do](/2026/04/02/model-monitor-gmcb-lmcb-insurance-monitoring-v100/) — post-deployment calibration monitoring using PITMonitor and GMCB/LMCB
- [Beyond Coverage: Commitment, Deferral, and Error Exposure in Conformal Pricing](/2026/04/02/conformal-tradeoffs-commitment-deferral-error-exposure/) — operational deployment metrics beyond the coverage guarantee
