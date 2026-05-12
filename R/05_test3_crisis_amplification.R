# ==============================================================================
# Conglomerate Paper - Empirical Section
# Script 05: Test 3 — Crisis amplification
#
# Question: under multiplicative dynamics, variance drag = sigma^2/2 becomes
# the binding constraint precisely when sigma is large. Pooling benefits
# should therefore widen during high-volatility periods, even if the
# unconditional firm-level effect is null (which Test 1 showed).
#
# This is the test of "dormant under normal conditions, activates in crises"
# — directly building on Kuppuswamy and Villalonga (2016).
#
# Three specifications:
#   3A: Binary crisis interaction (2001-02, 2008-09, 2020)
#       y = b1*Conglom + b2*Conglom:Crisis + controls + ind x year FE
#   3B: Continuous VIX interaction
#       y = b1*Conglom + b2*Conglom:VIX_z + controls + ind x year FE
#   3C: Within-conglomerate, bar_rho x Crisis
#       Does the pooling mechanism activate for low-correlation conglomerates
#       specifically during crises?
#
# Predictions:
#   3A, 3B:  b2 < 0 for sigma^2 and D, b2 > 0 for g
#   3C:      For sigma^2 in conglomerate-only sample, the bar_rho:Crisis
#            interaction should be positive (high-correlation conglomerates
#            can't pool, suffer more variance in crises)
#
# Author: Simon Blöthner
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------------------

PROJECT_DIR <- "/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE" 
RAW_DIR     <- file.path(PROJECT_DIR, "data", "raw")
PROC_DIR    <- file.path(PROJECT_DIR, "data", "processed")
OUT_DIR     <- file.path(PROJECT_DIR, "output", "test3")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

START_YEAR     <- 1998
END_YEAR       <- 2024
WINSORIZE_TAIL <- 0.01

CRISIS_YEARS <- c(2001, 2002, 2008, 2009, 2020)

required_packages <- c("dplyr", "tidyr", "purrr", "lubridate", "fixest")
new_pkg <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(new_pkg)) install.packages(new_pkg)
invisible(lapply(required_packages, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------------------

cat("Loading data...\n")

panel       <- readRDS(file.path(PROC_DIR, "firm_year_panel.rds"))
bar_rho_fy  <- readRDS(file.path(PROC_DIR, "bar_rho_per_firm_year.rds"))
vix_annual  <- readRDS(file.path(RAW_DIR,  "vix_annual.rds"))

cat(sprintf("  Firm-year panel:  %s rows\n",
            format(nrow(panel),       big.mark = ",")))
cat(sprintf("  VIX annual:       %s years\n",
            format(nrow(vix_annual),  big.mark = ",")))


# ------------------------------------------------------------------------------
# 3. Construct controls and merge crisis/VIX indicators
# ------------------------------------------------------------------------------

cat("\nConstructing controls and crisis indicators...\n")

# Standardized VIX (centered, scaled) for cleaner interaction interpretation
vix_for_merge <- vix_annual |>
  filter(year >= START_YEAR, year <= END_YEAR) |>
  mutate(
    vix_mean_z = (vix_mean - mean(vix_mean, na.rm = TRUE)) /
                 sd(vix_mean, na.rm = TRUE)
  ) |>
  rename(fyear = year) |>
  select(fyear, vix_mean, vix_mean_z, vix_q4)

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


# ------------------------------------------------------------------------------
# 4. Apply sample restrictions and winsorize
# ------------------------------------------------------------------------------

cat("\nPreparing regression sample...\n")

sample <- panel |>
  filter(fyear >= START_YEAR, fyear <= END_YEAR) |>
  filter(!is.na(mu_at_5y), !is.na(sigma2_at_5y)) |>
  filter(!is.na(sich2)) |>
  filter(at > 1, !is.na(log_at), !is.na(leverage),
         is.finite(leverage), leverage <= 2) |>
  filter(!is.na(vix_mean_z))

wins <- function(x, p = WINSORIZE_TAIL) {
  q <- quantile(x, c(p, 1 - p), na.rm = TRUE)
  pmin(pmax(x, q[1]), q[2])
}

sample <- sample |>
  mutate(
    mu_at_w     = wins(mu_at_5y),
    sigma2_at_w = wins(sigma2_at_5y),
    g_at_w      = wins(g_at_5y),
    D_at_w      = wins(D_at_5y),
    is_cong_num = as.integer(is_conglomerate)
  )

# For bar_rho-based specs, center within the conglomerate sample
mean_bar_rho <- mean(sample$bar_rho[sample$is_conglomerate], na.rm = TRUE)
sample <- sample |>
  mutate(bar_rho_c = bar_rho - mean_bar_rho)

cat(sprintf("  Final sample: %s firm-years\n",
            format(nrow(sample), big.mark = ",")))
cat(sprintf("  Crisis-year obs: %s (%.1f%%)\n",
            format(sum(sample$is_crisis), big.mark = ","),
            100 * mean(sample$is_crisis)))

# ------------------------------------------------------------------------------
# 5. Quick descriptive: sigma^2 and g by conglomerate status x crisis
# ------------------------------------------------------------------------------

cat("\n--- Descriptive: outcomes by conglomerate status and crisis era ---\n")

desc_cross <- sample |>
  group_by(is_conglomerate, is_crisis) |>
  summarise(
    n            = n(),
    mean_sigma2  = mean(sigma2_at_w, na.rm = TRUE),
    mean_g       = mean(g_at_w,      na.rm = TRUE),
    mean_D       = mean(D_at_w,      na.rm = TRUE),
    .groups = "drop"
  )
print(desc_cross)

# Implied diff-in-diff:
cat("\nImplied DiD for sigma^2 (Crisis effect on conglomerate vs standalone):\n")
sigma_did <- desc_cross |>
  select(is_conglomerate, is_crisis, mean_sigma2) |>
  pivot_wider(names_from = is_crisis, values_from = mean_sigma2,
              names_prefix = "crisis_") |>
  mutate(diff = crisis_TRUE - crisis_FALSE)
print(sigma_did)


# ------------------------------------------------------------------------------
# 6. Specification 3A: Binary crisis interaction
# ------------------------------------------------------------------------------

cat("\n--- Specification 3A: Binary crisis interaction ---\n")

m3a_sigma <- feols(
  sigma2_at_w ~ is_cong_num + is_cong_num:is_crisis +
                log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m3a_mu <- feols(
  mu_at_w ~ is_cong_num + is_cong_num:is_crisis +
            log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m3a_g <- feols(
  g_at_w ~ is_cong_num + is_cong_num:is_crisis +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m3a_D <- feols(
  D_at_w ~ is_cong_num + is_cong_num:is_crisis +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

etable(
  m3a_sigma, m3a_mu, m3a_g, m3a_D,
  headers = c("sigma^2", "mu", "g", "D"),
  fitstat = ~ n + r2 + war2
)


# ------------------------------------------------------------------------------
# 7. Specification 3B: Continuous VIX interaction
# ------------------------------------------------------------------------------
#
# y = a + b1*Conglom + b2*Conglom:VIX_z + controls + sich2 x fyear FE
#
# Same predictions as 3A but with continuous volatility measure.
# Note: only post-1990 obs have VIX, so sample is restricted accordingly.
# ------------------------------------------------------------------------------

cat("\n--- Specification 3B: Continuous VIX interaction ---\n")

m3b_sigma <- feols(
  sigma2_at_w ~ is_cong_num + is_cong_num:vix_mean_z +
                log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m3b_mu <- feols(
  mu_at_w ~ is_cong_num + is_cong_num:vix_mean_z +
            log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m3b_g <- feols(
  g_at_w ~ is_cong_num + is_cong_num:vix_mean_z +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m3b_D <- feols(
  D_at_w ~ is_cong_num + is_cong_num:vix_mean_z +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

etable(
  m3b_sigma, m3b_mu, m3b_g, m3b_D,
  headers = c("sigma^2", "mu", "g", "D"),
  fitstat = ~ n + r2 + war2
)


# ------------------------------------------------------------------------------
# 8. Specification 3C: Within-conglomerate bar_rho x Crisis
# ------------------------------------------------------------------------------
#
# Sample: conglomerate firm-years with valid bar_rho.
#
# y = b1*bar_rho_c + b2*bar_rho_c:Crisis + controls + sich2 x fyear FE
#
# Theoretical interpretation: if pooling activates during crises, then
# low-bar_rho conglomerates (genuinely uncorrelated segments) should
# benefit more than high-bar_rho conglomerates.
#
# Predictions:
#   b2 > 0 for sigma^2 (high-rho conglomerates have more crisis variance,
#                       i.e., the pooling protection is correlation-dependent
#                       and steeper during crises)
#   b2 < 0 for g       (low-rho conglomerates grow faster in crises)
# ------------------------------------------------------------------------------

cat("\n--- Specification 3C: Within-conglomerate bar_rho x crisis ---\n")

cong_sample <- sample |> filter(is_conglomerate, !is.na(bar_rho))

cat(sprintf("  Conglomerate-only sample: %s firm-years\n",
            format(nrow(cong_sample), big.mark = ",")))

m3c_sigma <- feols(
  sigma2_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
                log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

m3c_mu <- feols(
  mu_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
            log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

m3c_g <- feols(
  g_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

m3c_D <- feols(
  D_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

etable(
  m3c_sigma, m3c_mu, m3c_g, m3c_D,
  headers = c("sigma^2", "mu", "g", "D"),
  fitstat = ~ n + r2 + war2
)

# ------------------------------------------------------------------------------
# 8b. Clustering robustness for Spec 3C
# ------------------------------------------------------------------------------
#
# The bar_rho_c:is_crisisTRUE coefficient in 3C is precisely estimated under
# two-way clustering. With only 5 crisis years out of 27, year-clustering may
# not adequately reflect within-year cross-sectional correlation. Re-estimate
# with single-way clustering to gauge robustness.
# ------------------------------------------------------------------------------

cat("\n--- 3C clustering robustness on sigma^2 ---\n\n")

formula_3c <- sigma2_at_w ~ bar_rho_c + bar_rho_c:is_crisis +
              log_at + leverage + rd_intensity + age | sich2^fyear

m3c_2way <- feols(formula_3c, data = cong_sample,
                  cluster = ~ gvkey + fyear)
m3c_firm <- feols(formula_3c, data = cong_sample,
                  cluster = ~ gvkey)
m3c_year <- feols(formula_3c, data = cong_sample,
                  cluster = ~ fyear)
m3c_hc1  <- feols(formula_3c, data = cong_sample,
                  vcov = "hetero")

etable(m3c_2way, m3c_firm, m3c_year, m3c_hc1,
       headers = c("Two-way", "Firm only", "Year only", "Hetero (no cluster)"),
       fitstat = ~ n + r2)

# ------------------------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------------------------

cat("\nSaving regression objects...\n")

saveRDS(list(sigma2 = m3a_sigma, mu = m3a_mu, g = m3a_g, D = m3a_D),
        file.path(OUT_DIR, "test3a_crisis_dummy.rds"))
saveRDS(list(sigma2 = m3b_sigma, mu = m3b_mu, g = m3b_g, D = m3b_D),
        file.path(OUT_DIR, "test3b_vix_continuous.rds"))
saveRDS(list(sigma2 = m3c_sigma, mu = m3c_mu, g = m3c_g, D = m3c_D),
        file.path(OUT_DIR, "test3c_within_cong_bar_rho.rds"))


# ------------------------------------------------------------------------------
# 10. Headline summary
# ------------------------------------------------------------------------------

cat("\n--- Headline summary ---\n\n")

# Corrected summary loops — use "sigma" not "sigma2"
extract_interact <- function(mod, term) {
  cf <- coef(mod)
  se <- sqrt(diag(vcov(mod)))
  if (!term %in% names(cf)) return(c(NA, NA))
  c(cf[term], se[term])
}

# Corrected loops — use is_cong_num: terms
cat("\nSpec 3A (binary crisis): is_cong_num:is_crisisTRUE\n")
for (out_name in c("sigma", "mu", "g", "D")) {
  mod <- get(paste0("m3a_", out_name))
  res <- extract_interact(mod, "is_cong_num:is_crisisTRUE")
  cat(sprintf("  %-8s  b2 = %+.4f (s.e. %.4f)\n", out_name, res[1], res[2]))
}

cat("\nSpec 3B (continuous VIX): is_cong_num:vix_mean_z\n")
for (out_name in c("sigma", "mu", "g", "D")) {
  mod <- get(paste0("m3b_", out_name))
  res <- extract_interact(mod, "is_cong_num:vix_mean_z")
  cat(sprintf("  %-8s  b2 = %+.4f (s.e. %.4f)\n", out_name, res[1], res[2]))
}

cat("\nSpec 3C (within-conglomerate): bar_rho_c:is_crisisTRUE\n")
for (out_name in c("sigma", "mu", "g", "D")) {
  mod <- get(paste0("m3c_", out_name))
  res <- extract_interact(mod, "bar_rho_c:is_crisisTRUE")
  cat(sprintf("  %-8s  b2 = %+.4f (s.e. %.4f)\n", out_name, res[1], res[2]))
}

cat("\n--- Predictions reminder ---\n")
cat("  3A, 3B:  b2 < 0 for sigma^2 and D, b2 > 0 for g\n")
cat("  3C:      b2 > 0 for sigma^2, b2 < 0 for g\n")
cat("           (i.e., low-bar_rho conglomerates protect better in crises)\n")

cat("\n--- Done. Output saved to: ---\n")
cat(sprintf("  %s\n", normalizePath(OUT_DIR)))
