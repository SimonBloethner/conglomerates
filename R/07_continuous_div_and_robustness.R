# ==============================================================================
# Conglomerate Paper - Empirical Section
# Script 07: Continuous diversification reruns and identification robustness
#
# Two purposes:
#
# (A) Continuous BH version of previously binary tests:
#     - Test 1 (variance drag) was run with is_conglomerate dummy
#     - Test 3A/3B (crisis amplification) used is_conglomerate dummy
#     The binary dummy collapses the curvature in diversification depth.
#     Berry-Herfindahl gives the dose-response the model is actually about.
#
# (B) Identification robustness on the central Test 3C finding:
#     - Triple interaction with leverage to partially separate pooling from
#       coinsurance (Lewellen 1971): coinsurance benefits scale with leverage,
#       pooling should not
#     - Within-firm fixed effects: identify off changes in bar_rho within
#       firm over time, absorbing all time-invariant firm characteristics
#
# Author: Simon Blöthner
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------------------

PROJECT_DIR <- "/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE" 
RAW_DIR     <- file.path(PROJECT_DIR, "data", "raw")
PROC_DIR    <- file.path(PROJECT_DIR, "data", "processed")
OUT_DIR     <- file.path(PROJECT_DIR, "output", "test7")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

START_YEAR     <- 1998
END_YEAR       <- 2024
WINSORIZE_TAIL <- 0.01
CRISIS_YEARS   <- c(2001, 2002, 2008, 2009, 2020)

required_packages <- c("dplyr", "tidyr", "purrr", "lubridate", "fixest")
new_pkg <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(new_pkg)) install.packages(new_pkg)
invisible(lapply(required_packages, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 2. Load and prepare sample
# ------------------------------------------------------------------------------

cat("Loading data and preparing sample...\n")

panel       <- readRDS(file.path(PROC_DIR, "firm_year_panel.rds"))
bar_rho_fy  <- readRDS(file.path(PROC_DIR, "bar_rho_per_firm_year.rds"))
vix_annual  <- readRDS(file.path(RAW_DIR,  "vix_annual.rds"))

vix_for_merge <- vix_annual |>
  filter(year >= START_YEAR, year <= END_YEAR) |>
  mutate(vix_mean_z = (vix_mean - mean(vix_mean, na.rm = TRUE)) /
                     sd(vix_mean, na.rm = TRUE)) |>
  rename(fyear = year) |>
  select(fyear, vix_mean, vix_mean_z)

panel <- panel |>
  mutate(
    sich2        = suppressWarnings(as.integer(sich)) %/% 100,
    log_at       = log(at),
    leverage     = (coalesce(dlc, 0) + coalesce(dltt, 0)) / at,
    rd_intensity = coalesce(xrd, 0) / at,
    age          = pmax(0, fyear - lubridate::year(ipodate)),
    is_crisis    = fyear %in% CRISIS_YEARS
  ) |>
  left_join(bar_rho_fy, by = c("gvkey", "datadate")) |>
  left_join(vix_for_merge, by = "fyear")

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
  filter(!is.na(vix_mean_z)) |>
  mutate(
    mu_at_w     = wins(mu_at_5y),
    sigma2_at_w = wins(sigma2_at_5y),
    g_at_w      = wins(g_at_5y),
    D_at_w      = wins(D_at_5y),
    bh_div_w    = wins(bh_div),
    # Standardize BH and leverage for cleaner interaction interpretation
    bh_div_z    = (bh_div_w - mean(bh_div_w, na.rm = TRUE)) /
                   sd(bh_div_w, na.rm = TRUE),
    leverage_z  = (leverage  - mean(leverage,  na.rm = TRUE)) /
                   sd(leverage,  na.rm = TRUE)
  )

# Bar_rho centered within conglomerate sample (for the bar_rho-based specs)
mean_bar_rho <- mean(sample$bar_rho[sample$is_conglomerate], na.rm = TRUE)
sample <- sample |> mutate(bar_rho_c = bar_rho - mean_bar_rho)

cat(sprintf("  Final sample: %s firm-years\n",
            format(nrow(sample), big.mark = ",")))


# ==============================================================================
# Part A: Continuous BH reruns of binary tests
# ==============================================================================


# ------------------------------------------------------------------------------
# 3. Test 1 redux: BH (linear-only) replacing is_conglomerate
# ------------------------------------------------------------------------------
#
# Earlier (script 02): is_conglomerate on g gave -0.021 (significant)
# Test 6C with BH quadratic: linear +0.048, quadratic -0.146 (inverted-U)
# Here: BH linear only, no quadratic. This is the cleanest one-coefficient
# version of "how does diversification depth associate with outcomes?"
# ------------------------------------------------------------------------------

cat("\n--- Test 1 redux: linear BH (no quadratic) ---\n")

m1bh_sigma <- feols(
  sigma2_at_w ~ bh_div_z + log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m1bh_mu <- feols(
  mu_at_w ~ bh_div_z + log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m1bh_g <- feols(
  g_at_w ~ bh_div_z + log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m1bh_D <- feols(
  D_at_w ~ bh_div_z + log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)

etable(m1bh_sigma, m1bh_mu, m1bh_g, m1bh_D,
       headers = c("sigma^2", "mu", "g", "D"),
       fitstat = ~ n + r2 + war2)


# ------------------------------------------------------------------------------
# 4. Test 3A/3B redux: BH x Crisis and BH x VIX
# ------------------------------------------------------------------------------
#
# Earlier (script 05): is_conglomerate x Crisis on g gave +0.012 (marginal)
# Here: BH x Crisis. Does dose-response amplify the crisis interaction?
# ------------------------------------------------------------------------------

cat("\n--- Test 3 redux: BH x Crisis (full sample) ---\n")

m3a_bh_sigma <- feols(
  sigma2_at_w ~ bh_div_z + bh_div_z:is_crisis +
                log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m3a_bh_mu <- feols(
  mu_at_w ~ bh_div_z + bh_div_z:is_crisis +
            log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m3a_bh_g <- feols(
  g_at_w ~ bh_div_z + bh_div_z:is_crisis +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m3a_bh_D <- feols(
  D_at_w ~ bh_div_z + bh_div_z:is_crisis +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)

etable(m3a_bh_sigma, m3a_bh_mu, m3a_bh_g, m3a_bh_D,
       headers = c("sigma^2", "mu", "g", "D"),
       fitstat = ~ n + r2 + war2)


cat("\n--- Test 3B redux: BH x VIX (continuous) ---\n")

m3b_bh_sigma <- feols(
  sigma2_at_w ~ bh_div_z + bh_div_z:vix_mean_z +
                log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m3b_bh_g <- feols(
  g_at_w ~ bh_div_z + bh_div_z:vix_mean_z +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)
m3b_bh_D <- feols(
  D_at_w ~ bh_div_z + bh_div_z:vix_mean_z +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data = sample, cluster = ~ gvkey + fyear
)

etable(m3b_bh_sigma, m3b_bh_g, m3b_bh_D,
       headers = c("sigma^2", "g", "D"),
       fitstat = ~ n + r2 + war2)


# ==============================================================================
# Part B: Identification robustness on Test 3C
# ==============================================================================


# ------------------------------------------------------------------------------
# 5. Triple interaction with leverage (pooling vs coinsurance)
# ------------------------------------------------------------------------------
#
# Pooling (this paper): variance reduction operates through internal capital
# transfers, roughly leverage-invariant
# Coinsurance (Lewellen 1971): variance reduction operates through enhanced
# debt capacity, scales strongly with leverage
#
# Test: bar_rho_c x Crisis x leverage_z
#   If pooling is the channel: triple interaction should be small / null
#   If coinsurance is the channel: triple interaction should be strongly
#     positive on sigma^2 (high-leverage low-bar_rho conglomerates benefit
#     much more)
# ------------------------------------------------------------------------------

cat("\n--- Robustness 1: Triple interaction with leverage (pooling vs coinsurance) ---\n")

cong_sample <- sample |> filter(is_conglomerate, !is.na(bar_rho))

m3c_lev_sigma <- feols(
  sigma2_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
                bar_rho_c:leverage_z + bar_rho_c:is_crisis:leverage_z +
                leverage_z + leverage_z:is_crisis +
                log_at + rd_intensity + age | sich2^fyear,
  data = cong_sample, cluster = ~ gvkey
)

m3c_lev_g <- feols(
  g_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
          bar_rho_c:leverage_z + bar_rho_c:is_crisis:leverage_z +
          leverage_z + leverage_z:is_crisis +
          log_at + rd_intensity + age | sich2^fyear,
  data = cong_sample, cluster = ~ gvkey
)

etable(m3c_lev_sigma, m3c_lev_g,
       headers = c("sigma^2", "g"),
       fitstat = ~ n + r2 + war2)


# ------------------------------------------------------------------------------
# 6. Within-firm fixed effects on Test 3C
# ------------------------------------------------------------------------------
#
# Identification off changes in bar_rho within firm over time, absorbing
# all time-invariant firm characteristics. The sample shrinks because only
# firms with within-firm variation in bar_rho contribute to identification.
#
# Two specifications:
#   (a) firm FE + year FE (gives within-firm-over-time identification)
#   (b) firm FE + industry-year FE (the strongest absorption available)
# ------------------------------------------------------------------------------

cat("\n--- Robustness 2: Within-firm FE on Test 3C ---\n")

cat(sprintf("  Conglomerate-only sample for within-firm analysis: %s firm-years\n",
            format(nrow(cong_sample), big.mark = ",")))

# Quick sanity check: how much within-firm variation in bar_rho exists?
within_var <- cong_sample |>
  group_by(gvkey) |>
  summarise(
    n_obs       = n(),
    sd_bar_rho  = sd(bar_rho_c, na.rm = TRUE)
  ) |>
  filter(n_obs >= 2)
cat(sprintf("  Firms with 2+ observations: %s\n",
            format(nrow(within_var), big.mark = ",")))
cat(sprintf("  Median within-firm sd(bar_rho): %.3f\n",
            median(within_var$sd_bar_rho, na.rm = TRUE)))

m3c_ffe_sigma <- feols(
  sigma2_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
                log_at + leverage + rd_intensity + age | gvkey + fyear,
  data = cong_sample, cluster = ~ gvkey
)

m3c_ffe_g <- feols(
  g_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
          log_at + leverage + rd_intensity + age | gvkey + fyear,
  data = cong_sample, cluster = ~ gvkey
)

m3c_ffe_full_sigma <- feols(
  sigma2_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
                log_at + leverage + rd_intensity + age |
                gvkey + sich2^fyear,
  data = cong_sample, cluster = ~ gvkey
)

m3c_ffe_full_g <- feols(
  g_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
          log_at + leverage + rd_intensity + age |
          gvkey + sich2^fyear,
  data = cong_sample, cluster = ~ gvkey
)

etable(m3c_ffe_sigma, m3c_ffe_full_sigma, m3c_ffe_g, m3c_ffe_full_g,
       headers = c("sigma^2: firm+yr FE", "sigma^2: firm+indyr FE",
                   "g: firm+yr FE",       "g: firm+indyr FE"),
       fitstat = ~ n + r2 + war2)


# ------------------------------------------------------------------------------
# 7. Save and summary
# ------------------------------------------------------------------------------

saveRDS(list(
  bh_baseline   = list(sigma = m1bh_sigma, mu = m1bh_mu, g = m1bh_g, D = m1bh_D),
  bh_crisis     = list(sigma = m3a_bh_sigma, mu = m3a_bh_mu,
                       g = m3a_bh_g, D = m3a_bh_D),
  bh_vix        = list(sigma = m3b_bh_sigma, g = m3b_bh_g, D = m3b_bh_D),
  leverage_3way = list(sigma = m3c_lev_sigma, g = m3c_lev_g),
  firm_fe_3c    = list(sigma_y = m3c_ffe_sigma, sigma_iy = m3c_ffe_full_sigma,
                       g_y = m3c_ffe_g, g_iy = m3c_ffe_full_g)
), file.path(OUT_DIR, "test7_models.rds"))


cat("\n--- Headline summary ---\n\n")

cat("Test 1 redux (BH replaces binary):\n")
for (out_name in c("sigma", "mu", "g", "D")) {
  mod <- get(paste0("m1bh_", out_name))
  cf <- coef(mod); se <- sqrt(diag(vcov(mod)))
  cat(sprintf("  %-8s  bh_div_z = %+.4f (s.e. %.4f)\n",
              out_name, cf["bh_div_z"], se["bh_div_z"]))
}

cat("\nTest 3A redux (BH x Crisis):\n")
for (out_name in c("sigma", "mu", "g", "D")) {
  mod <- get(paste0("m3a_bh_", out_name))
  cf <- coef(mod); se <- sqrt(diag(vcov(mod)))
  term <- "bh_div_z:is_crisisTRUE"
  cat(sprintf("  %-8s  b2 = %+.4f (s.e. %.4f)\n",
              out_name, cf[term], se[term]))
}

cat("\nLeverage triple interaction (3C robustness):\n")
for (out_name in c("sigma", "g")) {
  mod <- get(paste0("m3c_lev_", out_name))
  cf <- coef(mod); se <- sqrt(diag(vcov(mod)))
  term <- "bar_rho_c:is_crisisTRUE:leverage_z"
  cat(sprintf("  %-8s  triple = %+.4f (s.e. %.4f)\n",
              out_name, cf[term], se[term]))
}

cat("\nFirm fixed effects (3C robustness):\n")
for (mod_name in c("m3c_ffe_sigma", "m3c_ffe_full_sigma",
                   "m3c_ffe_g", "m3c_ffe_full_g")) {
  mod <- get(mod_name)
  cf <- coef(mod); se <- sqrt(diag(vcov(mod)))
  term <- "bar_rho_c:is_crisisTRUE"
  cat(sprintf("  %-30s  %+.4f (s.e. %.4f)\n",
              mod_name, cf[term], se[term]))
}

cat("\n--- Done ---\n")
cat(sprintf("  Output saved to: %s\n", normalizePath(OUT_DIR)))
