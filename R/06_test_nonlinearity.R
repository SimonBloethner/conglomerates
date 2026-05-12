# ==============================================================================
# Conglomerate Paper - Empirical Section
# Script 06: Non-linearity test in conglomerate scope
#
# Tests an explicit prediction of the model (Section 4 and Figure 3):
# the relationship between conglomerate size and growth outcomes is
# non-monotonic — pooling benefits dominate at low N, coordination costs
# dominate at high N, producing an inverted-U on g.
#
# Three specifications:
#   6A: Quadratic in segment count
#         y = b1*N + b2*N^2 + controls + FE
#         Prediction for g: b1 > 0, b2 < 0
#         Prediction for sigma^2: b1 < 0, b2 ≥ 0
#
#   6B: Flexible (indicator for each segment count level)
#         y = sum_k I(N = k) * gamma_k + controls + FE
#         Let the data show the shape without imposing quadratic form
#
#   6C: Berry-Herfindahl as continuous diversification depth
#         y = b1*BH + b2*BH^2 + controls + FE
#
# Author: Simon Blöthner
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------------------

PROJECT_DIR <- "/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE" 
PROC_DIR    <- file.path(PROJECT_DIR, "data", "processed")
OUT_DIR     <- file.path(PROJECT_DIR, "output", "test_nonlin")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

START_YEAR     <- 1998
END_YEAR       <- 2024
WINSORIZE_TAIL <- 0.01
MAX_SEGMENTS_FLEX <- 8   # cap for flexible spec to avoid noisy bins

required_packages <- c("dplyr", "tidyr", "purrr", "lubridate", "fixest", "ggplot2")
new_pkg <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(new_pkg)) install.packages(new_pkg)
invisible(lapply(required_packages, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 2. Load and prepare
# ------------------------------------------------------------------------------

cat("Loading data...\n")
panel <- readRDS(file.path(PROC_DIR, "firm_year_panel.rds"))

panel <- panel |>
  mutate(
    sich2        = suppressWarnings(as.integer(sich)) %/% 100,
    log_at       = log(at),
    leverage     = (coalesce(dlc, 0) + coalesce(dltt, 0)) / at,
    rd_intensity = coalesce(xrd, 0) / at,
    age          = pmax(0, fyear - lubridate::year(ipodate))
  )

wins <- function(x, p = WINSORIZE_TAIL) {
  q <- quantile(x, c(p, 1 - p), na.rm = TRUE)
  pmin(pmax(x, q[1]), q[2])
}

sample <- panel |>
  filter(fyear >= START_YEAR, fyear <= END_YEAR) |>
  filter(!is.na(mu_at_5y), !is.na(sigma2_at_5y)) |>
  filter(!is.na(sich2)) |>
  filter(at > 1, !is.na(log_at), !is.na(leverage),
         is.finite(leverage), leverage <= 2) |>
  mutate(
    mu_at_w     = wins(mu_at_5y),
    sigma2_at_w = wins(sigma2_at_5y),
    g_at_w      = wins(g_at_5y),
    D_at_w      = wins(D_at_5y),
    # Topcode segment count to limit influence of rare very-high values
    n_seg       = pmin(unique_sic2, 15),
    n_seg_sq    = n_seg^2,
    bh_div_w    = wins(bh_div),
    bh_div_sq   = bh_div_w^2
  )

cat(sprintf("  Final sample: %s firm-years\n",
            format(nrow(sample), big.mark = ",")))

cat("\nDistribution of segment counts in sample:\n")
print(sample |>
  count(n_seg) |>
  mutate(pct = round(100 * n / sum(n), 2)))


# ------------------------------------------------------------------------------
# 3. Specification 6A: Quadratic in segment count
# ------------------------------------------------------------------------------

cat("\n--- Specification 6A: Quadratic in segment count ---\n")

m6a_sigma <- feols(
  sigma2_at_w ~ n_seg + n_seg_sq + log_at + leverage + rd_intensity + age |
                sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m6a_mu <- feols(
  mu_at_w ~ n_seg + n_seg_sq + log_at + leverage + rd_intensity + age |
            sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m6a_g <- feols(
  g_at_w ~ n_seg + n_seg_sq + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m6a_D <- feols(
  D_at_w ~ n_seg + n_seg_sq + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

etable(m6a_sigma, m6a_mu, m6a_g, m6a_D,
       headers = c("sigma^2", "mu", "g", "D"),
       fitstat = ~ n + r2 + war2)

# Implied optimal segment count from each model
implied_optimum <- function(mod, var_lin = "n_seg", var_sq = "n_seg_sq") {
  cf <- coef(mod)
  if (!(var_lin %in% names(cf) && var_sq %in% names(cf))) return(NA)
  if (cf[var_sq] == 0) return(NA)
  -cf[var_lin] / (2 * cf[var_sq])
}

cat("\nImplied turning points (vertex of the parabola in N):\n")
cat(sprintf("  sigma^2:  N* = %.2f  %s\n",
            implied_optimum(m6a_sigma),
            ifelse(coef(m6a_sigma)["n_seg_sq"] > 0, "(minimum)", "(maximum)")))
cat(sprintf("  mu:       N* = %.2f  %s\n",
            implied_optimum(m6a_mu),
            ifelse(coef(m6a_mu)["n_seg_sq"] > 0, "(minimum)", "(maximum)")))
cat(sprintf("  g:        N* = %.2f  %s\n",
            implied_optimum(m6a_g),
            ifelse(coef(m6a_g)["n_seg_sq"] > 0, "(minimum)", "(maximum)")))
cat(sprintf("  D:        N* = %.2f  %s\n",
            implied_optimum(m6a_D),
            ifelse(coef(m6a_D)["n_seg_sq"] > 0, "(minimum)", "(maximum)")))


# ------------------------------------------------------------------------------
# 4. Specification 6B: Flexible indicators for each segment count
# ------------------------------------------------------------------------------

cat("\n--- Specification 6B: Flexible (indicator per segment count level) ---\n")

flex_sample <- sample |> filter(n_seg <= MAX_SEGMENTS_FLEX)

m6b_sigma <- feols(
  sigma2_at_w ~ i(n_seg, ref = 1) + log_at + leverage + rd_intensity + age |
                sich2^fyear,
  data    = flex_sample,
  cluster = ~ gvkey + fyear
)

m6b_mu <- feols(
  mu_at_w ~ i(n_seg, ref = 1) + log_at + leverage + rd_intensity + age |
            sich2^fyear,
  data    = flex_sample,
  cluster = ~ gvkey + fyear
)

m6b_g <- feols(
  g_at_w ~ i(n_seg, ref = 1) + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = flex_sample,
  cluster = ~ gvkey + fyear
)

m6b_D <- feols(
  D_at_w ~ i(n_seg, ref = 1) + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = flex_sample,
  cluster = ~ gvkey + fyear
)

etable(m6b_sigma, m6b_mu, m6b_g, m6b_D,
       headers = c("sigma^2", "mu", "g", "D"),
       fitstat = ~ n + r2 + war2)


# Plot the flexible coefficients for g and sigma^2
plot_flex <- function(mod, var_name) {
  cf <- coef(mod)
  se <- sqrt(diag(vcov(mod)))
  keep <- grepl("^n_seg::", names(cf))
  if (!any(keep)) return(invisible(NULL))
  pts <- tibble(
    n_seg = as.integer(gsub("n_seg::", "", names(cf)[keep])),
    beta  = cf[keep],
    se    = se[keep]
  ) |>
    bind_rows(tibble(n_seg = 1, beta = 0, se = 0)) |>
    arrange(n_seg) |>
    mutate(lo = beta - 1.96 * se,
           hi = beta + 1.96 * se)

  ggplot(pts, aes(x = n_seg, y = beta)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2) +
    geom_point(size = 3) +
    geom_line() +
    scale_x_continuous(breaks = 1:MAX_SEGMENTS_FLEX) +
    labs(x = "Number of distinct 2-digit SIC segments (N)",
         y = sprintf("Coefficient on N (relative to N=1) for %s", var_name),
         title = sprintf("Flexible non-linearity: %s by segment count", var_name),
         subtitle = "95% confidence intervals; reference category: single-segment firms") +
    theme_minimal()
}

p_g     <- plot_flex(m6b_g,     "g (time-average growth rate)")
p_sigma <- plot_flex(m6b_sigma, "sigma^2 (growth variance)")
p_D     <- plot_flex(m6b_D,     "D (variance drag)")
p_mu    <- plot_flex(m6b_mu,    "mu (mean growth)")

ggsave(file.path(OUT_DIR, "nonlin_g.png"),     p_g,     width = 7, height = 5)
ggsave(file.path(OUT_DIR, "nonlin_sigma.png"), p_sigma, width = 7, height = 5)
ggsave(file.path(OUT_DIR, "nonlin_D.png"),     p_D,     width = 7, height = 5)
ggsave(file.path(OUT_DIR, "nonlin_mu.png"),    p_mu,    width = 7, height = 5)

cat(sprintf("\n  Flexible coefficient plots saved to: %s\n",
            normalizePath(OUT_DIR)))


# ------------------------------------------------------------------------------
# 5. Specification 6C: Berry-Herfindahl quadratic
# ------------------------------------------------------------------------------

cat("\n--- Specification 6C: Berry-Herfindahl diversification index ---\n")

m6c_sigma <- feols(
  sigma2_at_w ~ bh_div_w + bh_div_sq + log_at + leverage + rd_intensity + age |
                sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m6c_mu <- feols(
  mu_at_w ~ bh_div_w + bh_div_sq + log_at + leverage + rd_intensity + age |
            sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m6c_g <- feols(
  g_at_w ~ bh_div_w + bh_div_sq + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m6c_D <- feols(
  D_at_w ~ bh_div_w + bh_div_sq + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

etable(m6c_sigma, m6c_mu, m6c_g, m6c_D,
       headers = c("sigma^2", "mu", "g", "D"),
       fitstat = ~ n + r2 + war2)


# ------------------------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------------------------

saveRDS(list(quadratic_n = list(sigma2 = m6a_sigma, mu = m6a_mu,
                                 g = m6a_g, D = m6a_D),
             flexible_n  = list(sigma2 = m6b_sigma, mu = m6b_mu,
                                 g = m6b_g, D = m6b_D),
             quadratic_bh = list(sigma2 = m6c_sigma, mu = m6c_mu,
                                  g = m6c_g, D = m6c_D)),
        file.path(OUT_DIR, "test_nonlinearity_models.rds"))


# ------------------------------------------------------------------------------
# 7. Headline summary
# ------------------------------------------------------------------------------

cat("\n--- Headline summary ---\n\n")

extract_two <- function(mod, t1, t2) {
  cf <- coef(mod); se <- sqrt(diag(vcov(mod)))
  c(cf[t1], se[t1], cf[t2], se[t2])
}

cat("Spec 6A (quadratic in segment count):\n")
cat("                  b1 (linear)         b2 (quadratic)\n")
for (out_name in c("sigma", "mu", "g", "D")) {
  mod <- get(paste0("m6a_", out_name))
  r <- extract_two(mod, "n_seg", "n_seg_sq")
  cat(sprintf("  %-8s  %+.4f (%.4f)    %+.4f (%.4f)\n",
              out_name, r[1], r[2], r[3], r[4]))
}

cat("\nSpec 6C (quadratic in Berry-Herfindahl):\n")
cat("                  b1 (linear)         b2 (quadratic)\n")
for (out_name in c("sigma", "mu", "g", "D")) {
  mod <- get(paste0("m6c_", out_name))
  r <- extract_two(mod, "bh_div_w", "bh_div_sq")
  cat(sprintf("  %-8s  %+.4f (%.4f)    %+.4f (%.4f)\n",
              out_name, r[1], r[2], r[3], r[4]))
}

cat("\n--- Predictions reminder ---\n")
cat("  For g:        b1 > 0 and b2 < 0 (inverted-U: benefits then costs)\n")
cat("  For sigma^2:  b1 < 0 and b2 >= 0 (variance falls with N, diminishing)\n")
cat("  For D:        b1 < 0 and b2 >= 0\n")
cat("  For mu:       no theoretical prediction (model agnostic on mu vs N)\n")

cat("\n--- Done ---\n")
cat(sprintf("  Results saved to: %s\n", normalizePath(OUT_DIR)))
