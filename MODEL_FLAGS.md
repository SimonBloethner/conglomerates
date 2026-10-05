# Model Flags Documentation

## Growth Process Flags

### `growth_process`
Specifies how firm growth rates are generated.

Values:
- `"normal_net"` (default): Phase A baseline. `r ~ N(μ, σ²)`, then `log(1+r)` is stored.
- `"lognormal"`: `log δ = (μ - σ²/2) + σ·ε`, so `E[δ] = exp(μ)`.
- `"log_family"`: `log δ = μ + scale·ε` with family-specific innovations.

### `log_family`
Distribution family for innovations when `growth_process == "log_family"`.

Values:
- `"normal"` (default): Gaussian innovations.
- `"laplace"`: Laplace (double exponential) innovations.
- `"student_t"`: Student-t innovations with degrees of freedom specified by `nu`.

Only active when `growth_process == "log_family"`.

### `nu`
Degrees of freedom for Student-t distribution.

Default: `3.0`

Only active when `log_family == "student_t"`.

### `sigma_range`
Tuple `(min_sigma, max_sigma)` specifying bounds for volatility per market.

**Important:** Under `log_family`, `sigma_range` is interpreted as the **IQR (interquartile range)** of `log δ`, not the standard deviation. The scale parameter is computed from IQR using the distribution-specific conversion:
- Normal: `σ = IQR / (2 · Φ⁻¹(0.75)) ≈ IQR / 1.349`
- Laplace: `b = IQR / (2 · ln(2))`
- Student-t: `s = IQR / (2 · t⁻¹(0.75, df=ν))`

## Correlation Flags

### `rho`
Within-market correlation coefficient between firms.

Default: `0.0`

When `rho > 0` (or `rho < 0`), firms within the same market have correlated shocks using a factor model.

### `cross_corr`
Cross-market correlation coefficient.

Default: `0.0`

When `cross_corr > 0`, all pairs of firms across different markets have correlated shocks.

**Gaussian Copula:** When using a non-normal `log_family` with correlation (`rho > 0` or `cross_corr > 0`), a Gaussian copula is used:
1. Draw correlated standard normals using the existing correlation structure
2. Map through Φ (standard normal CDF) to get uniform marginals
3. Map through the family's quantile function

This preserves the marginal distributions while introducing the correlation structure.
