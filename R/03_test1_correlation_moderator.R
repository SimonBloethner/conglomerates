# ==============================================================================
# Conglomerate Paper - Empirical Section
# Script 03: Test 1 with cross-industry correlation moderator (continuous spec)
#
# Motivation: the baseline Test 1 with the binary is_conglomerate dummy
# returned a null result on variance reduction. The theory predicts the
# pooling benefit is sharpest when segments span uncorrelated industries.
# This script computes for each conglomerate firm-year the mean pairwise
# correlation across its segment industries (from Ken French monthly
# returns, rolling 60-month windows), and re-runs spec 1B with this
# continuous moderator.
#
# Predictions (using bar_rho as the across-segment correlation):
#   beta_Conglom:bar_rho > 0 for sigma^2 (pooling benefit weakens with rho)
#   beta_Conglom:bar_rho < 0 for g       (growth advantage weakens with rho)
#
# Byproduct: the industry-pair-year correlation panel is saved for reuse
# in Test 2 (cross-industry correlation and conglomerate formation).
#
# Author: Simon Blöthner
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------------------

PROJECT_DIR <- "/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE" 
RAW_DIR     <- file.path(PROJECT_DIR, "data", "raw")
PROC_DIR    <- file.path(PROJECT_DIR, "data", "processed")
OUT_DIR     <- file.path(PROJECT_DIR, "output", "test1")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

START_YEAR     <- 1998
END_YEAR       <- 2024
ROLL_MONTHS    <- 60        # trailing-60-month correlation window
WINSORIZE_TAIL <- 0.01

required_packages <- c("dplyr", "tidyr", "purrr", "lubridate",
                       "fixest", "modelsummary")
new_pkg <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(new_pkg)) install.packages(new_pkg)
invisible(lapply(required_packages, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------------------

cat("Loading data...\n")

panel        <- readRDS(file.path(PROC_DIR, "firm_year_panel.rds"))
segs         <- readRDS(file.path(PROC_DIR, "segment_year_panel.rds"))
kf_returns   <- readRDS(file.path(RAW_DIR,  "kf49_returns.rds"))
kf_sicmap    <- readRDS(file.path(RAW_DIR,  "kf49_sicmap.rds"))

cat(sprintf("  firm-year panel:    %s\n",
            format(nrow(panel),      big.mark = ",")))
cat(sprintf("  segment-year panel: %s\n",
            format(nrow(segs),       big.mark = ",")))
cat(sprintf("  KF49 returns:       %s industry-months\n",
            format(nrow(kf_returns), big.mark = ",")))


# ------------------------------------------------------------------------------
# 3. Build industry-pair-year correlation panel
# ------------------------------------------------------------------------------
#
# For each year T from 1980 to 2024, compute pairwise correlations between
# all 49 KF industries using monthly returns from months [T-4 Jan, T Dec]
# (60-month trailing window). Output: long-format panel of
# (year, ind_a, ind_b, rho) with ind_a < ind_b — i.e., upper triangle only,
# 1,176 unordered pairs per year.
# ------------------------------------------------------------------------------

cat("\nBuilding industry correlation panel (rolling 60-month windows)...\n")

# Lookup table: kf_code (e.g. "Agric") -> kf_ind (integer 1-49)
kf_lookup <- kf_sicmap |>
  distinct(kf_code, kf_ind) |>
  arrange(kf_ind)

# Pivot KF returns to wide format: rows = year-months, columns = industries
kf_wide <- kf_returns |>
  filter(kf_code %in% kf_lookup$kf_code) |>
  select(year, month, date, kf_code, ret) |>
  tidyr::pivot_wider(id_cols   = c(year, month, date),
                     names_from = kf_code,
                     values_from = ret) |>
  arrange(date)

# Compute pairwise correlation matrix for a single year's trailing window,
# return as long-format tibble of upper-triangle pairs
compute_year_corr <- function(yr, wide_data, lookup, min_obs = 48) {
  window <- wide_data |> filter(year >= yr - 4, year <= yr)
  if (nrow(window) < min_obs) return(NULL)

  ret_mat <- as.matrix(window[, lookup$kf_code])
  cor_mat <- cor(ret_mat, use = "pairwise.complete.obs")

  # Convert symmetric matrix to long format, keep upper triangle
  rownames(cor_mat) <- lookup$kf_ind
  colnames(cor_mat) <- lookup$kf_ind
  long <- as.data.frame(as.table(cor_mat))
  names(long) <- c("ind_a", "ind_b", "rho")
  long |>
    mutate(ind_a = as.integer(as.character(ind_a)),
           ind_b = as.integer(as.character(ind_b)),
           year  = yr) |>
    filter(ind_a < ind_b) |>
    select(year, ind_a, ind_b, rho)
}

years_to_compute <- 1980:END_YEAR
corr_panel <- purrr::map_dfr(years_to_compute,
                             compute_year_corr,
                             wide_data = kf_wide,
                             lookup    = kf_lookup)

cat(sprintf("  Correlation panel: %s industry-pair-year obs across %d years\n",
            format(nrow(corr_panel), big.mark = ","),
            length(unique(corr_panel$year))))

# Save for Test 2 reuse
saveRDS(corr_panel, file.path(PROC_DIR, "kf49_corr_panel.rds"))

# Sanity check: distribution of rho
cat("\nDistribution of pairwise industry correlations (all years):\n")
print(round(quantile(corr_panel$rho,
                     c(0.05, 0.25, 0.5, 0.75, 0.95), na.rm = TRUE), 3))


# ------------------------------------------------------------------------------
# 4. For each firm-year, get the set of KF49 industries it spans
# ------------------------------------------------------------------------------

cat("\nMapping segments to KF49 industries...\n")

# kf_ind already attached to segment_year_panel via attach_kf49 in script 01
# Aggregate to firm-year level: distinct industries
firm_year_kf <- segs |>
  filter(!is.na(kf_ind), kf_ind != 49) |>   # drop "Other" — uninformative
  mutate(year = lubridate::year(datadate)) |>
  group_by(gvkey, datadate, year) |>
  summarise(
    kf_inds = list(sort(unique(kf_ind))),
    n_kf49  = length(unique(kf_ind)),
    .groups = "drop"
  )

cat(sprintf("  %s firm-year obs with KF49-mapped segments\n",
            format(nrow(firm_year_kf), big.mark = ",")))
cat(sprintf("  %s firm-years span 2+ distinct KF49 industries\n",
            format(sum(firm_year_kf$n_kf49 >= 2), big.mark = ",")))


# ------------------------------------------------------------------------------
# 5. Compute mean pairwise correlation bar_rho per firm-year
# ------------------------------------------------------------------------------

cat("\nComputing bar_rho per firm-year...\n")

# Expand to long format: one row per (firm-year, industry-pair)
firm_year_pairs <- firm_year_kf |>
  filter(n_kf49 >= 2) |>
  mutate(pair_df = purrr::map(kf_inds, function(inds) {
    pairs <- utils::combn(inds, 2)
    tibble(ind_a = pairs[1, ], ind_b = pairs[2, ])
  })) |>
  select(gvkey, datadate, year, pair_df) |>
  tidyr::unnest(pair_df)

# Join with correlation panel
firm_year_pairs <- firm_year_pairs |>
  left_join(corr_panel, by = c("year", "ind_a", "ind_b"))

# Aggregate to firm-year: mean pairwise correlation
bar_rho_per_fy <- firm_year_pairs |>
  group_by(gvkey, datadate) |>
  summarise(
    bar_rho = mean(rho, na.rm = TRUE),
    n_pairs = sum(!is.na(rho)),
    .groups = "drop"
  ) |>
  filter(n_pairs > 0)

cat(sprintf("  Computed bar_rho for %s firm-years\n",
            format(nrow(bar_rho_per_fy), big.mark = ",")))
cat("\nDistribution of bar_rho across conglomerate firm-years:\n")
print(round(summary(bar_rho_per_fy$bar_rho), 3))


# ------------------------------------------------------------------------------
# 6. Merge bar_rho into firm-year panel and prepare regression sample
# ------------------------------------------------------------------------------

cat("\nPreparing regression sample...\n")

# Construct controls (mirror of script 02)
panel <- panel |>
  mutate(
    sich2        = suppressWarnings(as.integer(sich)) %/% 100,
    log_at       = log(at),
    leverage     = (coalesce(dlc, 0) + coalesce(dltt, 0)) / at,
    rd_intensity = coalesce(xrd, 0) / at,
    age          = pmax(0, fyear - lubridate::year(ipodate))
  ) |>
  left_join(bar_rho_per_fy, by = c("gvkey", "datadate"))

# Apply restrictions
sample <- panel |>
  filter(fyear >= START_YEAR, fyear <= END_YEAR) |>
  filter(!is.na(mu_at_5y), !is.na(sigma2_at_5y)) |>
  filter(!is.na(sich2)) |>
  filter(at > 1, !is.na(log_at), !is.na(leverage),
         is.finite(leverage), leverage <= 2)

# Winsorize outcomes
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

# Center bar_rho around its conglomerate-sample mean for cleaner interaction
# interpretation. The main effect of is_conglomerate then captures the effect
# at the average bar_rho, and the interaction captures how that effect changes
# per unit deviation from the average.
mean_bar_rho <- mean(sample$bar_rho[sample$is_conglomerate], na.rm = TRUE)
sample <- sample |>
  mutate(
    bar_rho_c          = bar_rho - mean_bar_rho,
    # Set to 0 for non-conglomerates so interaction term works
    bar_rho_c_for_int  = coalesce(bar_rho_c, 0)
  )

cat(sprintf("  Final regression sample: %s firm-years\n",
            format(nrow(sample), big.mark = ",")))
cat(sprintf("  Of which conglomerates with valid bar_rho: %s\n",
            format(sum(sample$is_conglomerate & !is.na(sample$bar_rho)),
                   big.mark = ",")))
cat(sprintf("  Mean bar_rho among conglomerates: %.3f (centered)\n",
            mean_bar_rho))


# ------------------------------------------------------------------------------
# 7. Diagnostics on bar_rho
# ------------------------------------------------------------------------------

cat("\n--- bar_rho diagnostics ---\n\n")

sample |>
  filter(is_conglomerate, !is.na(bar_rho)) |>
  left_join(firm_year_kf |> select(gvkey, datadate, n_kf49),
            by = c("gvkey", "datadate")) |>
  group_by(n_kf49) |>
  summarise(n = n(), mean_bar_rho = mean(bar_rho), .groups = "drop") |>
  print()


# ------------------------------------------------------------------------------
# 8. Continuous specification: full sample with interaction
# ------------------------------------------------------------------------------
#
# Model:  Y = a + b1*Conglom + b2*Conglom:bar_rho_c + controls + sich2 x fyear FE
#
# Interpretation:
#   b1: effect of being a conglomerate at the average bar_rho
#   b2: how that effect changes per unit change in bar_rho
#
# Predictions:
#   For sigma^2:    b1 < 0 (variance reduction at avg correlation),
#                   b2 > 0 (reduction weakens as correlation rises)
#   For mu:         b1, b2 not theoretically signed
#   For g:          b1 > 0 (growth advantage at avg correlation),
#                   b2 < 0 (advantage weakens as correlation rises)
#   For D:          b1 < 0, b2 > 0
# ------------------------------------------------------------------------------

cat("\n--- Continuous spec with interaction (full sample) ---\n")

m1c_sigma <- feols(
  sigma2_at_w ~ is_conglomerate + is_conglomerate:bar_rho_c_for_int +
                log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1c_mu <- feols(
  mu_at_w ~ is_conglomerate + is_conglomerate:bar_rho_c_for_int +
            log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1c_g <- feols(
  g_at_w ~ is_conglomerate + is_conglomerate:bar_rho_c_for_int +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

m1c_D <- feols(
  D_at_w ~ is_conglomerate + is_conglomerate:bar_rho_c_for_int +
          log_at + leverage + rd_intensity + age | sich2^fyear,
  data    = sample,
  cluster = ~ gvkey + fyear
)

etable(
  m1c_sigma, m1c_mu, m1c_g, m1c_D,
  headers = c("sigma^2", "mu", "g", "D"),
  fitstat = ~ n + r2 + war2
)


# ------------------------------------------------------------------------------
# 9. Conglomerate-only sample: bar_rho effect within the conglomerate group
# ------------------------------------------------------------------------------
#
# An alternative specification dropping non-conglomerates and using bar_rho
# directly. This gives a cleaner read on within-conglomerate heterogeneity:
# among conglomerates, do higher-correlation segment sets exhibit less
# variance reduction?
# ------------------------------------------------------------------------------

cat("\n--- Within-conglomerate spec (conglomerates only) ---\n")

cong_sample <- sample |> filter(is_conglomerate, !is.na(bar_rho))

cat(sprintf("  Conglomerate-only sample: %s firm-years\n",
            format(nrow(cong_sample), big.mark = ",")))

m1c_within_sigma <- feols(
  sigma2_at_w ~ bar_rho_c + log_at + leverage + rd_intensity + age |
                sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

m1c_within_mu <- feols(
  mu_at_w ~ bar_rho_c + log_at + leverage + rd_intensity + age |
            sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

m1c_within_g <- feols(
  g_at_w ~ bar_rho_c + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

m1c_within_D <- feols(
  D_at_w ~ bar_rho_c + log_at + leverage + rd_intensity + age |
          sich2^fyear,
  data    = cong_sample,
  cluster = ~ gvkey + fyear
)

etable(
  m1c_within_sigma, m1c_within_mu, m1c_within_g, m1c_within_D,
  headers = c("sigma^2", "mu", "g", "D"),
  fitstat = ~ n + r2 + war2
)


# ------------------------------------------------------------------------------
# 10. Save results
# ------------------------------------------------------------------------------

cat("\nSaving regression objects and bar_rho panel...\n")

saveRDS(bar_rho_per_fy, file.path(PROC_DIR, "bar_rho_per_firm_year.rds"))

saveRDS(list(sigma2 = m1c_sigma, mu = m1c_mu, g = m1c_g, D = m1c_D),
        file.path(OUT_DIR, "test1c_continuous_full.rds"))
saveRDS(list(sigma2 = m1c_within_sigma, mu = m1c_within_mu,
             g = m1c_within_g, D = m1c_within_D),
        file.path(OUT_DIR, "test1c_continuous_within.rds"))


# ------------------------------------------------------------------------------
# 11. Headline summary
# ------------------------------------------------------------------------------

cat("\n--- Headline summary ---\n\n")

cat("Full-sample interaction model:\n")
cat("  Y = b1*Conglom + b2*Conglom:bar_rho_c + ...\n")
cat("  b2: how the conglomerate effect changes per unit increase in correlation\n\n")

for (out_name in c("sigma2", "mu", "g", "D")) {
  mod <- get(paste0("m1c_", out_name))
  cf  <- coef(mod)
  se  <- sqrt(diag(vcov(mod)))
  ic_main  <- cf["is_conglomerateTRUE"]
  ic_int   <- cf["is_conglomerateTRUE:bar_rho_c_for_int"]
  se_main  <- se["is_conglomerateTRUE"]
  se_int   <- se["is_conglomerateTRUE:bar_rho_c_for_int"]
  cat(sprintf("  %-8s  b1 = %+.4f (s.e. %.4f)    b2 = %+.4f (s.e. %.4f)\n",
              out_name, ic_main, se_main, ic_int, se_int))
}

cat("\nWithin-conglomerate spec (effect of correlation among conglomerates):\n")
for (out_name in c("sigma2", "mu", "g", "D")) {
  mod <- get(paste0("m1c_within_", out_name))
  cf  <- coef(mod)
  se  <- sqrt(diag(vcov(mod)))
  cat(sprintf("  %-8s  b_rho = %+.4f (s.e. %.4f)\n",
              out_name, cf["bar_rho_c"], se["bar_rho_c"]))
}

cat("\n--- Done. Output saved to: ---\n")
cat(sprintf("  %s\n", normalizePath(OUT_DIR)))
cat(sprintf("\nCorrelation panel saved for Test 2 reuse:\n  %s\n",
            normalizePath(file.path(PROC_DIR, "kf49_corr_panel.rds"))))
