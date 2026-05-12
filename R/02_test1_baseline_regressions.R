# ==============================================================================
# Conglomerate Paper - Empirical Section
# Script 02: Test 1 baseline regressions
#
# Specifications:
#   1A: Variance drag on Conglomerate indicator (the headline test)
#   1B: Three-equation diagnostic — sigma^2, mu, g separately
#         to confirm the variance channel rather than the mean channel
#         drives the result
#
# Identification: industry x year fixed effects (2-digit SIC x fyear)
# Standard errors: two-way clustered by firm (gvkey) and year (fyear)
#
# Author: Simon Blöthner
# Last updated: May 2026
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------------------

PROJECT_DIR <- "/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE" 
PROC_DIR    <- file.path(PROJECT_DIR, "data", "processed")
OUT_DIR     <- file.path(PROJECT_DIR, "output", "test1")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Sample period and methodological choices
START_YEAR     <- 1998     # SFAS 131 era; pre-1998 as robustness
END_YEAR       <- 2024
MIN_ROLL_OBS   <- 4        # minimum non-NA observations in the 5-yr window
WINSORIZE_TAIL <- 0.01     # winsorize outcome variables at 1% / 99%

required_packages <- c(
  "dplyr", "tidyr", "purrr", "readr",
  "fixest",      # high-performance fixed-effects regressions
  "modelsummary" # optional, for clean output tables
)
new_pkg <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(new_pkg)) install.packages(new_pkg)
invisible(lapply(required_packages, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 2. Load and prepare panel
# ------------------------------------------------------------------------------

cat("Loading firm-year panel...\n")
panel <- readRDS(file.path(PROC_DIR, "firm_year_panel.rds"))
cat(sprintf("  loaded: %s rows, %d variables\n",
            format(nrow(panel), big.mark = ","), ncol(panel)))


# ------------------------------------------------------------------------------
# 3. Construct controls
# ------------------------------------------------------------------------------

cat("\nConstructing control variables...\n")

panel <- panel |>
  mutate(
    # 2-digit SIC for industry classification
    sich2         = suppressWarnings(as.integer(sich)) %/% 100,
    sich3         = suppressWarnings(as.integer(sich)) %/% 10,
    # Log size
    log_at        = log(at),
    # Leverage: total debt / total assets
    leverage      = (coalesce(dlc, 0) + coalesce(dltt, 0)) / at,
    # R&D intensity (treat missing R&D as zero — common convention)
    rd_intensity  = coalesce(xrd, 0) / at,
    # Firm age in years (from IPO if available; fall back to first-in-sample)
    age           = pmax(0, fyear - lubridate::year(ipodate)),
    # Number of segments (continuous diversification depth; primary measure
    # is the conglomerate dummy, but we'll use this for sensitivity below)
    n_seg_cont    = unique_sic2
  )


# ------------------------------------------------------------------------------
# 4. Apply sample restrictions
# ------------------------------------------------------------------------------

cat("\nApplying sample restrictions...\n")

# Track sample attrition for transparency
n_step <- function(d, label) {
  cat(sprintf("  %-45s %s rows\n", label,
              format(nrow(d), big.mark = ",")))
  invisible(d)
}

sample <- panel |>
  n_step("starting panel") |>

  # SFAS 131 era
  filter(fyear >= START_YEAR, fyear <= END_YEAR) |>
  n_step(sprintf("after fyear in [%d, %d]", START_YEAR, END_YEAR)) |>

  # Valid rolling moments (need both mu and sigma2 for D and g)
  filter(!is.na(mu_at_5y), !is.na(sigma2_at_5y)) |>
  n_step("after valid 5-yr moments") |>

  # Valid industry classification
  filter(!is.na(sich2)) |>
  n_step("after valid SIC2") |>

  # Drop firms with non-positive controls
  filter(at > 1, !is.na(log_at), !is.na(leverage),
         is.finite(leverage), leverage <= 2) |>  # leverage > 2 are data errors
  n_step("after positive size and sane leverage")

cat(sprintf("\nFinal sample: %s firm-years, %d unique firms\n",
            format(nrow(sample), big.mark = ","),
            length(unique(sample$gvkey))))


# ------------------------------------------------------------------------------
# 5. Winsorize outcome variables (1% / 99%)
# ------------------------------------------------------------------------------

cat(sprintf("\nWinsorizing outcomes at %.0f%% / %.0f%%...\n",
            WINSORIZE_TAIL * 100, (1 - WINSORIZE_TAIL) * 100))

wins <- function(x, p = WINSORIZE_TAIL) {
  q <- quantile(x, c(p, 1 - p), na.rm = TRUE)
  pmin(pmax(x, q[1]), q[2])
}

sample <- sample |>
  mutate(
    mu_at_w     = wins(mu_at_5y),
    sigma2_at_w = wins(sigma2_at_5y),
    g_at_w      = wins(g_at_5y),
    D_at_w      = wins(D_at_5y)
  )


# ------------------------------------------------------------------------------
# 6. Descriptive comparison after restrictions
# ------------------------------------------------------------------------------

cat("\n--- Descriptive comparison (post-restriction, post-winsorization) ---\n")

desc <- sample |>
  group_by(is_conglomerate) |>
  summarise(
    n              = n(),
    mean_mu        = mean(mu_at_w,     na.rm = TRUE),
    mean_sigma2    = mean(sigma2_at_w, na.rm = TRUE),
    mean_g         = mean(g_at_w,      na.rm = TRUE),
    mean_D         = mean(D_at_w,      na.rm = TRUE),
    mean_log_at    = mean(log_at,      na.rm = TRUE),
    mean_leverage  = mean(leverage,    na.rm = TRUE)
  )
print(desc)


# ------------------------------------------------------------------------------
# 7. Specification 1A: Variance drag (the headline)
# ------------------------------------------------------------------------------
#
# Strategy: build up the fixed-effect structure progressively so we can see
# how much of the raw gap is absorbed by industry composition, time trends,
# and industry x year interactions.
#
# Predictions: beta_1 < 0 (conglomerates have lower variance drag)
# ------------------------------------------------------------------------------

cat("\n--- Specification 1A: Variance drag ---\n")

controls_formula <- ~ is_conglomerate + log_at + leverage + rd_intensity + age

m1a_naive <- feols(
  D_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1a_indFE <- feols(
  D_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age | sich2,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1a_yrFE  <- feols(
  D_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age | fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1a_both  <- feols(
  D_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age |
           sich2 + fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1a_indyr <- feols(
  D_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age |
           sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

etable(
  m1a_naive, m1a_indFE, m1a_yrFE, m1a_both, m1a_indyr,
  headers = c("OLS", "+ Ind FE", "+ Yr FE", "+ Both", "+ Ind x Yr"),
  fitstat = ~ n + r2 + war2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)


# ------------------------------------------------------------------------------
# 8. Specification 1B: Three-equation diagnostic
# ------------------------------------------------------------------------------
#
# Use the strongest FE structure from 1A (industry x year) and run the same
# specification with three different outcomes. The key diagnostic question:
# does the variance reduction (beta_sigma2 < 0) drive the variance-drag result,
# or is there also a mean penalty (beta_mu < 0) that complicates the story?
#
# Predictions:
#   beta_sigma2 < 0  (conglomerates have lower growth variance)
#   beta_mu ≈ 0      (mean growth roughly unchanged — variance channel only)
#   beta_g > 0       (time-average growth rate is higher)
# ------------------------------------------------------------------------------

cat("\n--- Specification 1B: Three-equation diagnostic ---\n")

m1b_sigma <- feols(
  sigma2_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age |
                sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1b_mu <- feols(
  mu_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age |
            sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1b_g <- feols(
  g_at_w ~ is_conglomerate + log_at + leverage + rd_intensity + age |
           sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

# Include the variance drag from 1A for comparison
etable(
  m1b_sigma, m1b_mu, m1b_g, m1a_indyr,
  headers = c("sigma^2", "mu", "g", "D = mu - g"),
  fitstat = ~ n + r2 + war2,
  signif.code = c("***" = 0.01, "**" = 0.05, "*" = 0.10)
)


# ------------------------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------------------------

cat("\nSaving regression objects...\n")

results_1a <- list(
  naive  = m1a_naive,
  ind_fe = m1a_indFE,
  yr_fe  = m1a_yrFE,
  both   = m1a_both,
  ind_yr = m1a_indyr
)
results_1b <- list(
  sigma2 = m1b_sigma,
  mu     = m1b_mu,
  g      = m1b_g,
  D      = m1a_indyr
)

saveRDS(results_1a, file.path(OUT_DIR, "test1a_variance_drag.rds"))
saveRDS(results_1b, file.path(OUT_DIR, "test1b_three_equation.rds"))
saveRDS(sample,     file.path(OUT_DIR, "test1_sample.rds"))


# ------------------------------------------------------------------------------
# 10. Headline summary
# ------------------------------------------------------------------------------

cat("\n--- Headline summary ---\n\n")

# Pull the key coefficients into a clean text summary
coef_1a <- coef(m1a_indyr)["is_conglomerateTRUE"]
se_1a   <- sqrt(diag(vcov(m1a_indyr)))["is_conglomerateTRUE"]

coef_sigma <- coef(m1b_sigma)["is_conglomerateTRUE"]
se_sigma   <- sqrt(diag(vcov(m1b_sigma)))["is_conglomerateTRUE"]

coef_mu    <- coef(m1b_mu)["is_conglomerateTRUE"]
se_mu      <- sqrt(diag(vcov(m1b_mu)))["is_conglomerateTRUE"]

coef_g     <- coef(m1b_g)["is_conglomerateTRUE"]
se_g       <- sqrt(diag(vcov(m1b_g)))["is_conglomerateTRUE"]

cat("Industry x year FE, two-way clustered by firm and year:\n\n")
cat(sprintf("  Variance drag D:   %+.4f (s.e. %.4f)\n",   coef_1a,    se_1a))
cat(sprintf("  Variance sigma^2:  %+.4f (s.e. %.4f)\n",   coef_sigma, se_sigma))
cat(sprintf("  Mean mu:           %+.4f (s.e. %.4f)\n",   coef_mu,    se_mu))
cat(sprintf("  Growth rate g:     %+.4f (s.e. %.4f)\n",   coef_g,     se_g))

cat("\n--- Done. Regression objects saved to: ---\n")
cat(sprintf("  %s\n", normalizePath(OUT_DIR)))
