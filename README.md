# SMD — Statistical Manifold Dynamics

Research prototype modeling financial markets as a latent state moving on a curved statistical (Riemannian) manifold, where local **curvature** — not just volatility — is treated as a measure of structural fragility.

**Hypothesis:** predictable, high-magnitude price moves come from high geometric curvature (structural fragility), not just high variance. Regime changes occur when the market's state is forced through highly-curved regions of its information manifold.

## Status

Research prototype. No transaction costs, slippage, execution modeling, or live risk controls. Not investment advice.

## Contents

- [`MATH.md`](./MATH.md) — full mathematical formulation, explained from first principles (Fisher Information Metric, natural gradient flow, the Riemannian SDE, the Geodesic Tension alpha signal, worked numeric example)

## Approach at a glance

1. Encode raw market data into a latent state $\theta_t$
2. Learn the local metric tensor $g_\phi(\theta_t)$ (via Cholesky decomposition, for validity)
3. Integrate a Riemannian SDE forward in time (natural gradient drift + geometry-scaled noise)
4. Compute local curvature and geodesic velocity
5. Combine curvature and driving force into the Geodesic Tension signal
6. Feed the signal into a standard portfolio optimizer for position sizing

## Evaluation

Benchmarked against Linear Regression, XGBoost, LSTM, and Transformer baselines on identical features, using purged walk-forward validation, Information Coefficient, turnover/slippage, and regime-specific (high-volatility/crash) performance.

## License

TBD.
