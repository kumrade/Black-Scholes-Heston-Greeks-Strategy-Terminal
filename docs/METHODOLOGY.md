# Calculation conventions and validation priorities

## Evidence scope

This write-up is based on the supplied v6.2 calculation code and ten analytical screenshots. It documents the mathematics and observed behavior. The original application and its external data-access code are not included in this documentation package.

Code blocks in the README are explanatory excerpts or pseudocode. They are not a runnable module or a supported public interface.

## Units

| Quantity | Convention in inspected implementation |
| --- | --- |
| `S`, `K`, premium | Price units per underlying unit |
| `T` | Years, using a 365-day calendar convention |
| `sigma`, `r`, `q` | Decimals in pricing functions |
| Displayed IV | Percentage units |
| IV residual | `abs(model_iv - market_iv) * 10000` with decimal IVs |
| BSM Vega | `dV/dsigma / 100` |
| Heston displayed Vega | `dV/dv0 / 100` |
| BSM Theta | Annual time derivative divided by 365 |
| Heston Theta | One-day repricing difference |
| BSM Rho / Heston Rho | Rate derivative divided by 100 |

BSM Vanna differentiates spot and volatility; the Heston implementation differentiates spot and initial variance. Their raw values are not directly comparable. The current Volga fields also have different coordinates and should not be used as squared percentage-point shock coefficients without conversion.

## Calibration sequence

1. Convert market IV percentages into decimals.
2. Retain finite IVs between 3% and 150%, strike/spot between 0.8 and 1.2, and raw BSM Vega above 0.05.
3. Require at least five valid observations.
4. Fit a robust quadratic in log-moneyness and blend noisy observations toward the smooth target.
5. Optionally remove large residual outliers if at least five points remain.
6. Normalize Vega-based weights adjusted for robustness.
7. Optimize weighted model-IV squared error plus parameter/Feller penalties.
8. Combine multi-start L-BFGS-B, differential evolution and final local refinement.
9. Report errors against raw retained market IVs.

The algorithm’s smoothed-target fitting objective differs from its raw-market error diagnostic. Both should be reported so smoothing does not conceal meaningful residuals.

## Snapshot parameters

| Parameter | Displayed value |
| --- | ---: |
| `kappa` | 0.1647 |
| `theta` | 0.07414 |
| `xi` | 0.12531 |
| `rho` | +0.16042 |
| `v0` | 0.05787 |

For this rounded vector, `sqrt(v0)` is approximately 24.06%, while `sqrt(theta)` is approximately 27.23%. Neither is the full strike-dependent Heston-implied smile. The Feller sufficient condition holds: `2*kappa*theta ≈ 0.02442 > xi² ≈ 0.01570`. That does not certify parameter uniqueness or fit quality.

## Price and IV residuals

The seven visible IV pairs imply a mean absolute residual of approximately 60.3 IV basis points, with a maximum around 126.4 basis points. These calculations use rounded screenshots, not a new calibration run.

A small price residual can generate a large IV residual when Vega is small. Locally:

```python
iv_error_decimal = price_error / vega_raw
```

This linear approximation requires a stable inverse and should not replace an actual IV solve.

## FFT resolution

At `N=4096`, `eta=0.05`, log-strike spacing is `2*pi/(N*eta) ≈ 0.0307`. Near spot 1040, this corresponds to roughly 32 spot units between grid strikes, wider than many 10-unit contract intervals in the screenshots.

Price interpolation, clipping and finite differences can therefore contribute to irregular smiles and unstable curvature. Compare against an independent pricer, converge frequency/strike settings, and sweep spot/variance bumps before interpreting Gamma or higher-order risk.

The optimizer compares candidate objective values from different FFT resolutions. Candidate selection should use the same final resolution for every candidate.

## Strategy scenarios

The implementation uses a common user-entered volatility and BSM theoretical entries for each option leg. Current, seven-day and approximate-expiry curves are hypothetical repricings. Exact expiry requires intrinsic payoff rather than rounding remaining days. Contracts need lot sizes and all relevant costs before per-unit P&L becomes position-level money exposure.

The heatmap is a spot-versus-time grid. Rebuilding should clear the full figure or remove the previous colorbar before adding another one.

## Attribution

The attribution tab selects the nearest-to-spot call, rather than the selected strategy portfolio. Its bar-chart total includes the volatility shock; its spot-shock curve omits that term. The curve and the total should use identical shocks or carry different labels.

Taylor terms omit rate risk, Vanna, Volga and other higher-order contributions. Full repricing should be used as a benchmark and the unexplained residual shown.

## Validation priorities

- Check BSM price-to-IV-to-price round trips and model-consistent price bounds.
- Check European put–call parity under common inputs.
- Compare Heston prices with an independent quadrature or other benchmark implementation.
- Verify price and Greek stability under FFT-grid, damping and bump changes.
- Retain independent raw CE and PE IVs alongside the selected calibration IV.
- Use exact expiry timestamps and explicit valuation conventions.
- Replace all unsupported strategy fallbacks with explicit rejection or correct mappings.
- Validate strike ordering, quantities, cash/stock overlays and per-leg expiries.
- Reconcile scenario charts and attribution against full repricing.
- Evaluate parameter stability on multiple expiries and unseen observations.

These are proposed checks, not tests that this documentation package claims to have passed.
