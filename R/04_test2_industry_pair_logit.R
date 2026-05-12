# ==============================================================================
# Conglomerate Paper - Empirical Section
# Script 04: Test 2 — Cross-industry correlation and conglomerate formation
#
# Two specifications:
#   2A: Descriptive null comparison
#       For each conglomerate firm-year, compare its mean pairwise segment
#       correlation (bar_rho) to the expected value under size-weighted
#       random industry sampling. Tests whether observed conglomerates
#       systematically form across less-correlated industries than chance.
#
#   2B: Industry-pair-year logit
#       Unit: KF49 industry pair x year. Dependent: indicator for any
#       conglomerate spanning the pair. Predicts conglomerate formation
#       from pair-level correlation, controlling for industry size and
#       characteristics. The strongest formal test of the theory's
#       distinguishing prediction.
#
# Predictions:
#   2A: mean(bar_rho - E[bar_rho | random]) < 0 across conglomerates
#   2B: beta_rho < 0 in the logit
#
# Author: Simon Blöthner
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Setup
# ------------------------------------------------------------------------------

PROJECT_DIR <- "/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE" 
RAW_DIR     <- file.path(PROJECT_DIR, "data", "raw")
PROC_DIR    <- file.path(PROJECT_DIR, "data", "processed")
OUT_DIR     <- file.path(PROJECT_DIR, "output", "test2")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

START_YEAR <- 1998
END_YEAR   <- 2024

required_packages <- c("dplyr", "tidyr", "purrr", "lubridate",
                       "fixest", "ggplot2")
new_pkg <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(new_pkg)) install.packages(new_pkg)
invisible(lapply(required_packages, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------------------

cat("Loading data...\n")

panel        <- readRDS(file.path(PROC_DIR, "firm_year_panel.rds"))
segs         <- readRDS(file.path(PROC_DIR, "segment_year_panel.rds"))
corr_panel   <- readRDS(file.path(PROC_DIR, "kf49_corr_panel.rds"))
bar_rho_fy   <- readRDS(file.path(PROC_DIR, "bar_rho_per_firm_year.rds"))

cat(sprintf("  Correlation panel: %s pair-years\n",
            format(nrow(corr_panel), big.mark = ",")))
cat(sprintf("  bar_rho per firm-year: %s obs\n",
            format(nrow(bar_rho_fy), big.mark = ",")))


# ------------------------------------------------------------------------------
# 3. Build industry-year statistics
# ------------------------------------------------------------------------------
#
# For each (KF49 industry, year), compute:
#   - n_firms_in_ind: distinct firms reporting any segment in that industry
#   - n_segments_in_ind: number of segment-year observations
#   - prob_ind: industry's share of all observed segments that year
#                (used as the size weight in the null distribution)
# ------------------------------------------------------------------------------

cat("\nBuilding industry-year statistics...\n")

ind_year_stats <- segs |>
  filter(!is.na(kf_ind), kf_ind != 49) |>   # drop "Other" catch-all
  mutate(year = lubridate::year(datadate)) |>
  filter(year >= START_YEAR, year <= END_YEAR) |>
  group_by(year, kf_ind) |>
  summarise(
    n_firms_in_ind    = n_distinct(gvkey),
    n_segments_in_ind = n(),
    total_ias         = sum(ias, na.rm = TRUE),
    .groups = "drop"
  ) |>
  group_by(year) |>
  mutate(prob_ind = n_firms_in_ind / sum(n_firms_in_ind)) |>
  ungroup()

cat(sprintf("  Industry-year cells: %s\n",
            format(nrow(ind_year_stats), big.mark = ",")))


# ------------------------------------------------------------------------------
# 4. Specification 2A: descriptive null comparison
# ------------------------------------------------------------------------------
#
# Question: are observed conglomerates more uncorrelated than random
# industry combinations would predict?
#
# Method: for each year, compute the expected pairwise correlation under
# size-weighted random sampling of two industries. Compare to the observed
# mean bar_rho across conglomerates in that year.
#
# Expected rho under random sampling (weighted by industry size):
#   E[rho | random] = sum_{j<k} 2*p_j*p_k*rho_jk / sum_{j<k} 2*p_j*p_k
#                   = sum_{j<k} 2*p_j*p_k*rho_jk / (1 - sum_j p_j^2)
# ------------------------------------------------------------------------------

cat("\n--- Specification 2A: Descriptive null comparison ---\n")

# For each year, compute the expected rho under random size-weighted sampling
exp_rho_by_year <- corr_panel |>
  filter(year >= START_YEAR, year <= END_YEAR) |>
  inner_join(ind_year_stats |>
               select(year, ind_a = kf_ind, prob_a = prob_ind),
             by = c("year", "ind_a")) |>
  inner_join(ind_year_stats |>
               select(year, ind_b = kf_ind, prob_b = prob_ind),
             by = c("year", "ind_b")) |>
  group_by(year) |>
  summarise(
    exp_rho_random = sum(prob_a * prob_b * rho) / sum(prob_a * prob_b),
    n_pairs_in_yr  = n(),
    .groups = "drop"
  )

# Observed mean bar_rho per year
obs_rho_by_year <- bar_rho_fy |>
  mutate(year = lubridate::year(datadate)) |>
  filter(year >= START_YEAR, year <= END_YEAR) |>
  group_by(year) |>
  summarise(
    obs_mean_bar_rho = mean(bar_rho, na.rm = TRUE),
    n_conglomerates  = n(),
    .groups = "drop"
  )

# Combine
yr_compare <- obs_rho_by_year |>
  inner_join(exp_rho_by_year, by = "year") |>
  mutate(delta = obs_mean_bar_rho - exp_rho_random)

cat("\nYear-by-year comparison (observed vs random-sampled expectation):\n")
print(yr_compare |>
        select(year, n_conglomerates, obs_mean_bar_rho,
               exp_rho_random, delta) |>
        mutate(across(c(obs_mean_bar_rho, exp_rho_random, delta), \(x) round(x, 3))))

# Pooled test: across all conglomerate firm-years, compute deviation
# from each year's expected random rho. Test mean(delta) < 0.
ind_delta <- bar_rho_fy |>
  mutate(year = lubridate::year(datadate)) |>
  filter(year >= START_YEAR, year <= END_YEAR) |>
  inner_join(exp_rho_by_year |> select(year, exp_rho_random),
             by = "year") |>
  mutate(delta = bar_rho - exp_rho_random)

cat("\nPooled test (one-sided H1: observed < random):\n")
pooled_test <- t.test(ind_delta$delta,
                      alternative = "less",
                      mu = 0)
print(pooled_test)

cat(sprintf("\n  Mean delta: %+.4f\n", mean(ind_delta$delta, na.rm = TRUE)))
cat(sprintf("  Median delta: %+.4f\n", median(ind_delta$delta, na.rm = TRUE)))
cat(sprintf("  N conglomerate firm-years: %s\n",
            format(sum(!is.na(ind_delta$delta)), big.mark = ",")))


# ------------------------------------------------------------------------------
# 5. Build industry-pair-year panel for Specification 2B
# ------------------------------------------------------------------------------

cat("\nBuilding industry-pair-year panel...\n")

# Observed conglomerate-spanned pairs: for each firm-year with 2+ KF49 industries,
# enumerate all pairs spanned by that firm
firm_year_kf <- segs |>
  filter(!is.na(kf_ind), kf_ind != 49) |>
  mutate(year = lubridate::year(datadate)) |>
  filter(year >= START_YEAR, year <= END_YEAR) |>
  group_by(gvkey, datadate, year) |>
  summarise(kf_inds = list(sort(unique(kf_ind))),
            n_kf49  = length(unique(kf_ind)),
            .groups = "drop") |>
  filter(n_kf49 >= 2)

observed_pairs <- firm_year_kf |>
  mutate(pair_df = purrr::map(kf_inds, function(inds) {
    pairs <- utils::combn(inds, 2)
    tibble(ind_a = pairs[1, ], ind_b = pairs[2, ])
  })) |>
  select(year, pair_df) |>
  tidyr::unnest(pair_df) |>
  group_by(year, ind_a, ind_b) |>
  summarise(n_conglomerates_spanning = n(),
            .groups = "drop") |>
  mutate(has_conglomerate = 1L)

# Full pair-year grid
all_inds <- sort(unique(ind_year_stats$kf_ind))
pair_grid <- expand.grid(
  ind_a = all_inds,
  ind_b = all_inds,
  year  = START_YEAR:END_YEAR
) |>
  as_tibble() |>
  filter(ind_a < ind_b)

# Merge correlations, observed indicator, and industry-level controls
pair_panel <- pair_grid |>
  inner_join(corr_panel, by = c("year", "ind_a", "ind_b")) |>
  left_join(observed_pairs, by = c("year", "ind_a", "ind_b")) |>
  mutate(
    has_conglomerate           = coalesce(has_conglomerate, 0L),
    n_conglomerates_spanning   = coalesce(n_conglomerates_spanning, 0L)
  ) |>
  # Industry-level controls
  left_join(ind_year_stats |>
              select(year, ind_a = kf_ind,
                     n_firms_a = n_firms_in_ind, prob_a = prob_ind),
            by = c("year", "ind_a")) |>
  left_join(ind_year_stats |>
              select(year, ind_b = kf_ind,
                     n_firms_b = n_firms_in_ind, prob_b = prob_ind),
            by = c("year", "ind_b")) |>
  filter(!is.na(n_firms_a), !is.na(n_firms_b)) |>
  mutate(
    log_n_a    = log(n_firms_a),
    log_n_b    = log(n_firms_b),
    log_n_avg  = (log_n_a + log_n_b) / 2,
    pair_size  = log(prob_a * prob_b)   # joint size (log)
  )

cat(sprintf("  Pair-year panel: %s observations\n",
            format(nrow(pair_panel), big.mark = ",")))
cat(sprintf("  Of which has_conglomerate = 1: %s (%.1f%%)\n",
            format(sum(pair_panel$has_conglomerate), big.mark = ","),
            100 * mean(pair_panel$has_conglomerate)))


# ------------------------------------------------------------------------------
# 6. Specification 2B: Industry-pair-year logit
# ------------------------------------------------------------------------------

cat("\n--- Specification 2B: Industry-pair-year logit ---\n")

# Sensitivity to fixed-effect structure
m2_pooled <- feglm(
  has_conglomerate ~ rho + log_n_a + log_n_b,
  data    = pair_panel,
  family  = binomial("logit"),
  cluster = ~ year
)

m2_yr <- feglm(
  has_conglomerate ~ rho + log_n_a + log_n_b | year,
  data    = pair_panel,
  family  = binomial("logit"),
  cluster = ~ year
)

m2_ind <- feglm(
  has_conglomerate ~ rho + log_n_a + log_n_b | ind_a + ind_b,
  data    = pair_panel,
  family  = binomial("logit"),
  cluster = ~ year
)

m2_full <- feglm(
  has_conglomerate ~ rho + log_n_a + log_n_b | ind_a + ind_b + year,
  data    = pair_panel,
  family  = binomial("logit"),
  cluster = ~ year
)

# Also test with count of conglomerates spanning (Poisson)
m2_pois <- fepois(
  n_conglomerates_spanning ~ rho + log_n_a + log_n_b |
                             ind_a + ind_b + year,
  data    = pair_panel,
  cluster = ~ year
)

cat("\nLogit and Poisson specifications:\n")
etable(
  m2_pooled, m2_yr, m2_ind, m2_full, m2_pois,
  headers = c("OLS-Logit", "+ Yr FE", "+ Ind FE", "+ Full FE", "Poisson count"),
  fitstat = ~ n + pr2 + ll
)


# ------------------------------------------------------------------------------
# 7. LPM specification as alternative
# ------------------------------------------------------------------------------

cat("\n--- Linear probability model (for marginal-effect interpretation) ---\n")

m2_lpm <- feols(
  has_conglomerate ~ rho + log_n_a + log_n_b |
                     ind_a + ind_b + year,
  data    = pair_panel,
  cluster = ~ year
)

etable(m2_lpm, fitstat = ~ n + r2 + war2)


# ------------------------------------------------------------------------------
# 8. Save outputs
# ------------------------------------------------------------------------------

cat("\nSaving outputs...\n")

saveRDS(pair_panel,     file.path(PROC_DIR, "pair_year_panel.rds"))
saveRDS(yr_compare,     file.path(OUT_DIR,  "test2a_yearly.rds"))
saveRDS(ind_delta,      file.path(OUT_DIR,  "test2a_ind_deltas.rds"))
saveRDS(list(pooled = m2_pooled, yr_fe = m2_yr, ind_fe = m2_ind,
             full = m2_full, pois = m2_pois, lpm = m2_lpm),
        file.path(OUT_DIR, "test2b_logit_models.rds"))


# ------------------------------------------------------------------------------
# 9. Headline summary
# ------------------------------------------------------------------------------

cat("\n--- Headline summary ---\n\n")

cat("Specification 2A — descriptive null comparison:\n")
cat(sprintf("  Mean observed bar_rho:        %.3f\n",
            mean(bar_rho_fy$bar_rho, na.rm = TRUE)))
cat(sprintf("  Mean expected (random size-w): %.3f\n",
            mean(yr_compare$exp_rho_random)))
cat(sprintf("  Pooled mean delta:             %+.4f\n",
            mean(ind_delta$delta, na.rm = TRUE)))
cat(sprintf("  t-statistic (H1: delta < 0):   %.2f\n",
            pooled_test$statistic))
cat(sprintf("  p-value (one-sided):           %.4g\n",
            pooled_test$p.value))

cat("\nSpecification 2B — pair-year logit (full FE, ind_a + ind_b + year):\n")
cf <- coef(m2_full)
se <- sqrt(diag(vcov(m2_full)))
cat(sprintf("  beta on rho:  %+.4f (s.e. %.4f)\n",
            cf["rho"], se["rho"]))
cat(sprintf("  Sample: %s pair-year observations\n",
            format(m2_full$nobs, big.mark = ",")))

cat("\n--- Done. Output saved to: ---\n")
cat(sprintf("  %s\n", normalizePath(OUT_DIR)))
