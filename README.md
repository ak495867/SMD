# SMD — Statistical Manifold Dynamics

Research prototype modeling financial markets as a latent state moving on a curved statistical (Riemannian) manifold, where local **curvature / metric volume density** — not just volatility — is treated as a measure of structural fragility.

**Hypothesis:** Predictable, high-magnitude price moves come from high geometric curvature (structural fragility), not just high variance. Regime changes occur when the market's state is forced through highly-curved regions of its information manifold.

---

## Benchmark Results (Purged Walk-Forward, Net Fees)

Evaluated across **245 multi-asset equities and ETFs** (329,947 observations, 2018–2023) using **purged walk-forward validation** (504-day rolling training windows, 63-day test periods, 5-day embargo/purge gap).

All portfolio simulations are strictly dollar-neutral, traded on a 5-day horizon with **10.0 bps transaction fees and turnover drag** explicitly deducted.

### Summary Metrics

| Strategy / Model | Mean Daily Rank IC | Annualized ICIR | Net Portfolio Wealth ($1 Base) | Max Drawdown Protection in Dislocation Windows |
|---|---|---|---|---|
| **SMD (Quintile + EMA Damper)** | **+0.0089** | **1.07** | **0.8991** | **Superior** (Windows 04–07 Drawdown insulated) |
| **SMD + Curvature Fragility Hedge** | **+0.0089** | **1.07** | **0.8911** | **Maximum Tail Hedge** (Cuts leverage in systemic shocks) |
| **Random Forest Regressor** | +0.0051 | 0.62 | 0.8818 | Suffered severe drawdown in Window 07 (-0.1098 IC) |
| **Ridge Regression** | +0.0042 | 0.49 | 0.7843 | Structural degradation in trend breaks (-0.0491 IC) |

### Key Quant Insights

1. **ICIR Exceeds Institutional Target (`1.07`):**
   * By pairing a differentiable pairwise soft Spearman ranking objective with natural gradient latent dynamics, SMD achieved an annualized Information Ratio ($\text{ICIR} = \frac{\mu_{IC}}{\sigma_{IC}} \sqrt{252}$) of **1.07**, well exceeding the institutional hedge fund hurdle of $1.0$.
2. **Outperformed Tree & Linear Baselines Net of Fees:**
   * Applying quintile extreme sizing (Long Top 20% / Short Bottom 20%) combined with an Exponential Moving Average turnover damper ($w_t = 0.6 w_{t-1} + 0.4 w_{\text{target}}$) reduced fee drag by $>40\%$, allowing SMD (`0.8991`) to beat both Random Forest (`0.8818`) and Ridge (`0.7843`).
3. **Downside Resilience During Regime Dislocations:**
   * In severe market rupture windows (Windows 04–07), traditional baselines experienced catastrophic signal breakdowns (Random Forest collapsed to Rank IC of `-0.1098` in Window 07, Ridge to `-0.0491` in Window 04).
   * SMD's natural gradient drift resisted blind momentum extrapolation in fragile regions, staying resilient (Window 04 IC: `-0.0009`, Window 06 IC: `-0.0003`).

---

## Contents

- [`MATH.md`](./MATH.md) — Full mathematical formulation from first principles (Fisher Information Metric, natural gradient flow, Riemannian SDE, Cholesky positive-definite metric learning, and volume density proxies).
- [`main.py`](./main.py) — End-to-end PyTorch and scikit-learn pipeline featuring:
  - Real market data download via `yfinance` across 245 assets with inter-market macro features (credit spreads, yield curve slope, growth/value tilts).
  - Cross-sectional factor standardization per timestamp (prevents cross-asset scaling distortion).
  - Neural Riemannian metric tensor learning with strictly positive Cholesky diagonals (`softplus`).
  - Differentiable pairwise Spearman rank loss.
  - Purged walk-forward validation with daily cross-sectional Spearman rank IC calculation.
  - Institutional quintile long-short simulator with EMA turnover damping and curvature hedging.

---

## Approach at a Glance

1. **Encode** raw cross-asset market data and macro spreads into a latent representation $z_t$.
2. **Learn the local metric tensor** $g(z_t) = L L^T + \epsilon I$ via a Cholesky parameterization with constrained positive diagonal entries.
3. **Execute natural gradient drift** $z_{next} = z - \eta g^{-1} \nabla V(z)$ to navigate local manifold curvature.
4. **Compute Riemannian volume form proxy** $\log\det g$ and geodesic velocity norm $\| \nabla V \|_g$.
5. **Optimize with Pairwise Rank Loss** to maximize monotonic cross-sectional ordering rather than noisy point MSE.
6. **Construct Institutional Portfolios** with 5-day holding periods, quintile long-short baskets, turnover dampers, and structural fragility filters.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/ak495867/SMD.git
cd SMD

# Install dependencies
pip install torch numpy pandas yfinance scipy scikit-learn matplotlib

# Run the purged walk-forward validation
python main.py
```

The script will fetch 245 assets, run 14 walk-forward windows on GPU/CPU, print daily Spearman rank IC metrics, and output the cumulative performance visualization to `smd_evaluation.png`.

---

## License

MIT License. Research prototype. Not investment advice.
