# ==============================================================================
# Conglomerate Paper - Empirical Section
# One-stop data acquisition script
#
# Pulls:
#   1. Compustat Fundamentals Annual (comp.funda)            [WRDS]
#   2. Compustat Historical Segments (comp.wrds_segmerged)   [WRDS]
#   3. (Optional) CRSP/Compustat link table                  [WRDS]
#   4. Ken French 49 Industry Portfolios (monthly returns)   [Dartmouth]
#   5. Ken French SIC-to-Industry-49 classification          [Dartmouth]
#   6. VIX daily series                                      [FRED]
#
# Outputs analysis-ready RDS files in <project>/data/
# Author: Simon Blöthner
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------

PROJECT_DIR <- "/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE"                        # change if running from elsewhere
RAW_DIR     <- file.path(PROJECT_DIR, "data", "raw")
PROC_DIR    <- file.path(PROJECT_DIR, "data", "processed")
START_YEAR  <- 1980
END_YEAR    <- 2024

dir.create(RAW_DIR,  recursive = TRUE, showWarnings = FALSE)
dir.create(PROC_DIR, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------------
# 2. Package management
# ------------------------------------------------------------------------------

required_packages <- c(
  "DBI", "RPostgres",                                    # WRDS connection
  "dplyr", "tidyr", "purrr", "readr", "stringr",         # tidyverse core
  "lubridate",                                           # dates
  "zoo",                                                 # rolling windows
  "frenchdata"                                           # Ken French data
)
new_pkg <- setdiff(required_packages, installed.packages()[, "Package"])
if (length(new_pkg)) install.packages(new_pkg)
invisible(lapply(required_packages, library, character.only = TRUE))


# ------------------------------------------------------------------------------
# 3. WRDS connection
# ------------------------------------------------------------------------------
#
# WRDS uses PostgreSQL on wrds-pgdata.wharton.upenn.edu:9737/wrds.
# Two authentication options:
#
# (a) Recommended: ~/.pgpass file with line
#        wrds-pgdata.wharton.upenn.edu:9737:wrds:USERNAME:PASSWORD
#     then chmod 0600 ~/.pgpass. No interactive prompts after that.
#
# (b) Interactive prompts (this script's fallback).
#
# Reference: https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-r/r-from-your-computer/

get_wrds_creds <- function() {
  user <- Sys.getenv("WRDS_USER")
  pw   <- Sys.getenv("WRDS_PW")
  if (user == "") {
    if (interactive()) {
      user <- readline("WRDS username: ")
    } else {
      stop("Set WRDS_USER environment variable or run interactively.")
    }
  }
  if (pw == "") {
    if (interactive()) {
      pw <- if (requireNamespace("rstudioapi", quietly = TRUE) &&
                rstudioapi::isAvailable()) {
        rstudioapi::askForPassword("WRDS password")
      } else {
        readline("WRDS password (visible): ")
      }
    } else {
      stop("Set WRDS_PW environment variable or run interactively.")
    }
  }
  list(user = user, pw = pw)
}

cat("Connecting to WRDS...\n")
creds <- get_wrds_creds()
wrds <- dbConnect(
  Postgres(),
  host     = "wrds-pgdata.wharton.upenn.edu",
  port     = 9737,
  dbname   = "wrds",
  sslmode  = "require",
  user     = creds$user,
  password = creds$pw
)
rm(creds)  # don't keep password in workspace

# Sanity check
conn_info <- dbGetQuery(wrds, "SELECT current_user, current_database();")
cat(sprintf("Connected as %s to database %s.\n",
            conn_info$current_user, conn_info$current_database))

# Check subscription access to Historical Segments before doing heavy pulls
seg_access <- tryCatch(
  dbGetQuery(wrds, "SELECT * FROM comp.wrds_segmerged LIMIT 1;"),
  error = function(e) NULL
)
if (is.null(seg_access)) {
  warning("Cannot read comp.wrds_segmerged. Your WRDS subscription may not ",
          "include Compustat Historical Segments. Resolve this before proceeding.")
} else {
  cat("Access to comp.wrds_segmerged confirmed.\n")
}
rm(seg_access)


# ------------------------------------------------------------------------------
# 4. PULL 1: Compustat Fundamentals Annual
# ------------------------------------------------------------------------------

cat("\n[Pull 1/3] Compustat Fundamentals Annual...\n")

# ---- Compustat Fundamentals (time-series variables only) ----
funda_query <- sprintf("
  SELECT gvkey, conm, fyear, datadate,
         at, sale, oibdp, ebit, ni, xsga, xrd, capx, emp,
         dlc, dltt, ceq, prcc_f, csho,
         sich, naicsh, exchg, fic
  FROM comp.funda
  WHERE indfmt = 'INDL'
    AND datafmt = 'STD'
    AND popsrc  = 'D'
    AND consol  = 'C'
    AND curcd   = 'USD'
    AND fyear BETWEEN %d AND %d
    AND at IS NOT NULL AND at > 0;
", START_YEAR, END_YEAR)

funda <- dbGetQuery(wrds, funda_query) |> as_tibble()
cat(sprintf("  funda: %s rows\n", format(nrow(funda), big.mark = ",")))

# ---- Compustat Company (static header variables) ----
# Note: fic is in both funda and company; we use funda's and skip it here
company_query <- "
  SELECT gvkey, loc, state, dlrsn, dldte, ipodate,
         sic AS sic_curr, naics AS naics_curr
  FROM comp.company;
"
company <- dbGetQuery(wrds, company_query) |> as_tibble()
cat(sprintf("  company: %s firms\n", format(nrow(company), big.mark = ",")))

# ---- Merge ----
funda <- funda |> left_join(company, by = "gvkey")
saveRDS(funda, file.path(RAW_DIR, "funda_raw.rds"))
cat(sprintf("  merged firm-year panel: %s rows, %d variables\n",
            format(nrow(funda), big.mark = ","), ncol(funda)))


# ------------------------------------------------------------------------------
# 5. PULL 2: Compustat Historical Segments
# ------------------------------------------------------------------------------

cat("\n[Pull 2/3] Compustat Historical Segments (full panel)...\n")

seg_query <- sprintf("
  SELECT gvkey, datadate, srcdate, stype, sid, snms,
         sics1, sics2, naicss1,
         sales, ias, ops, emps, capxs
  FROM comp_segments_hist_daily.wrds_segmerged
  WHERE stype IN ('BUSSEG', 'OPSEG')
    AND datadate BETWEEN '%d-01-01' AND '%d-12-31';
", START_YEAR, END_YEAR)

t0 <- Sys.time()
segments_raw <- dbGetQuery(wrds, seg_query) |> as_tibble()
cat(sprintf("  %s rows in %.1fs\n",
            format(nrow(segments_raw), big.mark = ","),
            as.numeric(Sys.time() - t0, units = "secs")))

# Deduplicate: when the same gvkey-datadate-sid appears multiple times due
# to restatements (srcdate > datadate), keep the as-first-reported version
# (earliest srcdate). This is the standard convention in published work.
segments <- segments_raw |>
  group_by(gvkey, datadate, sid, stype) |>
  slice_min(srcdate, n = 1, with_ties = FALSE) |>
  ungroup()

cat(sprintf("  After dedup: %s rows\n",
            format(nrow(segments), big.mark = ",")))

saveRDS(segments, file.path(RAW_DIR, "segments_raw.rds"))


# ------------------------------------------------------------------------------
# 6. PULL 3 (optional): CRSP/Compustat link
# ------------------------------------------------------------------------------
# Uncomment if you want to add stock-return-based volatility as robustness.
#
# cat("\n[Pull 3/3] CRSP/Compustat Merged link table...\n")
# ccm_query <- "
  # SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
  # FROM crsp.ccmxpf_lnkhist
  # WHERE linktype IN ('LC', 'LU')
    # AND linkprim IN ('P', 'C');
# "
# ccm <- dbGetQuery(wrds, ccm_query) |> as_tibble()
# saveRDS(ccm, file.path(RAW_DIR, "ccm_link.rds"))
# cat(sprintf("  %s rows\n", format(nrow(ccm), big.mark = ",")))


# Disconnect from WRDS now that pulls are done
dbDisconnect(wrds)
cat("\nWRDS connection closed.\n")


# ------------------------------------------------------------------------------
# 7. Ken French 49 Industry Portfolios (monthly returns)
# ------------------------------------------------------------------------------

cat("\nDownloading Ken French 49 Industry Portfolios...\n")

# frenchdata wraps the Dartmouth files and handles parsing
kf_raw <- download_french_data("49 Industry Portfolios")

# The package returns a list with multiple subtables. We want value-weighted
# monthly returns (the standard for industry-correlation work).
kf49_vw <- kf_raw$subsets$data[[
  which(kf_raw$subsets$name == "Average Value Weighted Returns -- Monthly")
]]

# Reshape: long format (date, kf_ind_code, ret)
kf49_returns <- kf49_vw |>
  rename(yearmonth = date) |>
  pivot_longer(-yearmonth, names_to = "kf_code", values_to = "ret_pct") |>
  mutate(
    yearmonth = as.integer(yearmonth),
    year  = yearmonth %/% 100,
    month = yearmonth %% 100,
    date  = as.Date(sprintf("%d-%02d-01", year, month)),
    # KF reports as percent; convert to decimal
    ret   = ret_pct / 100,
    # KF uses -99.99 for missing
    ret   = if_else(ret_pct <= -99, NA_real_, ret)
  ) |>
  select(date, year, month, kf_code, ret)

saveRDS(kf49_returns, file.path(RAW_DIR, "kf49_returns.rds"))
cat(sprintf("  %s industry-month observations\n",
            format(nrow(kf49_returns), big.mark = ",")))


# ------------------------------------------------------------------------------
# 8. Ken French SIC-to-Industry-49 classification
# ------------------------------------------------------------------------------

cat("\nDownloading Ken French SIC-to-KF49 classification...\n")

kf_sic_url <- "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes49.zip"
zip_tmp <- tempfile(fileext = ".zip")
download.file(kf_sic_url, zip_tmp, mode = "wb", quiet = TRUE)
txt_file <- unzip(zip_tmp, exdir = tempdir())
sic_lines <- readLines(txt_file)

# Parse: industry header lines look like " 1 Agric  Agriculture"
#        SIC range lines look like "          0100-0199 Agricultural ..."
parse_kf49_mapping <- function(lines) {
  out <- list()
  cur <- list(ind = NA_integer_, code = NA_character_, name = NA_character_)
  for (line in lines) {
    if (nchar(trimws(line)) == 0) next
    # Header: digits at start, then code, then name
    m_header <- regmatches(line,
                regexec("^\\s*(\\d{1,2})\\s+(\\S+)\\s*(.*)$", line))[[1]]
    is_header <- length(m_header) == 4 &&
                 !grepl("\\d{4}-\\d{4}", line) &&
                 nchar(m_header[2]) <= 2
    if (is_header) {
      cur$ind  <- as.integer(m_header[2])
      cur$code <- m_header[3]
      cur$name <- trimws(m_header[4])
    } else {
      # SIC range line
      m_sic <- regmatches(line,
              regexec("^\\s+(\\d{4})-(\\d{4})", line))[[1]]
      if (length(m_sic) == 3) {
        out[[length(out) + 1]] <- tibble(
          kf_ind  = cur$ind,
          kf_code = cur$code,
          kf_name = cur$name,
          sic_lo  = as.integer(m_sic[2]),
          sic_hi  = as.integer(m_sic[3])
        )
      }
    }
  }
  bind_rows(out)
}

kf49_ranges <- parse_kf49_mapping(sic_lines)

# Expand to one row per 4-digit SIC for fast joining later
kf49_sicmap <- kf49_ranges |>
  rowwise() |>
  mutate(sic4_list = list(seq(sic_lo, sic_hi))) |>
  unnest(sic4_list) |>
  rename(sic4 = sic4_list) |>
  ungroup() |>
  select(sic4, kf_ind, kf_code, kf_name) |>
  distinct(sic4, .keep_all = TRUE)  # first matching range wins (none should overlap)

saveRDS(kf49_sicmap,  file.path(RAW_DIR, "kf49_sicmap.rds"))
saveRDS(kf49_ranges,  file.path(RAW_DIR, "kf49_ranges.rds"))
cat(sprintf("  %d unique 4-digit SIC codes mapped\n", nrow(kf49_sicmap)))


# ------------------------------------------------------------------------------
# 9. VIX from FRED
# ------------------------------------------------------------------------------

cat("\nDownloading VIX from FRED...\n")
vix_url <- "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
tmp <- tempfile(fileext = ".csv")
download.file(vix_url, tmp, mode = "wb", quiet = TRUE,
              method = "libcurl")
vix_daily <- read_csv(tmp, show_col_types = FALSE) |>
  rename(date = 1, vix = 2) |>
  mutate(vix = suppressWarnings(as.numeric(vix))) |>
  filter(!is.na(vix))
unlink(tmp)

# Annual mean and top-quartile dummy
vix_annual <- vix_daily |>
  mutate(year = year(date)) |>
  group_by(year) |>
  summarise(vix_mean = mean(vix, na.rm = TRUE),
            vix_max  = max(vix,  na.rm = TRUE),
            .groups  = "drop") |>
  mutate(vix_q4 = vix_mean >= quantile(vix_mean, 0.75, na.rm = TRUE))

saveRDS(vix_daily,  file.path(RAW_DIR, "vix_daily.rds"))
saveRDS(vix_annual, file.path(RAW_DIR, "vix_annual.rds"))
cat(sprintf("  VIX %d daily obs, %d annual obs\n",
            nrow(vix_daily), nrow(vix_annual)))


# ==============================================================================
# 10. Build analysis-ready firm-year panel
# ==============================================================================

cat("\nBuilding firm-year panel with derived variables...\n")

# Helper: map a 4-digit SIC to KF49 industry; default 49 (Other) if no match
attach_kf49 <- function(df, sic_col) {
  df |>
    mutate(sic4 = suppressWarnings(as.integer({{ sic_col }}))) |>
    left_join(kf49_sicmap, by = "sic4") |>
    mutate(
      kf_ind  = if_else(is.na(kf_ind), 49L, kf_ind),
      kf_code = if_else(is.na(kf_code), "Other", kf_code)
    )
}

# Sample exclusions on funda
funda_clean <- funda |>
  mutate(
    sic_int = suppressWarnings(as.integer(sich)),
    is_financial = sic_int >= 6000 & sic_int <= 6999,
    is_utility   = sic_int >= 4900 & sic_int <= 4999,
    is_govt      = sic_int >= 9000 & sic_int <= 9999
  ) |>
  filter(!is_financial, !is_utility, !is_govt) |>
  attach_kf49(sic_int) |>
  select(-is_financial, -is_utility, -is_govt)

# Derived variables per firm-year from segments
seg_for_panel <- segments |>
  mutate(sic_int = suppressWarnings(as.integer(sics1))) |>
  attach_kf49(sic_int) |>
  mutate(year = year(datadate))

# For each firm-year, distinct SIC2 count, segment count, BH index, KF49 count
firm_year_seg_agg <- seg_for_panel |>
  filter(!is.na(ias), ias > 0) |>
  group_by(gvkey, datadate) |>
  summarise(
    n_segments        = n_distinct(sid),
    unique_sic2       = n_distinct(sic_int %/% 100, na.rm = TRUE),
    unique_sic3       = n_distinct(sic_int %/% 10,  na.rm = TRUE),
    unique_kf49       = n_distinct(kf_ind,          na.rm = TRUE),
    sum_ias           = sum(ias),
    bh_div            = 1 - sum((ias / sum(ias))^2),
    # capture the set of KF49 industries the firm spans (for Test 2)
    kf49_set          = paste(sort(unique(kf_ind)), collapse = ","),
    .groups = "drop"
  )

# Merge into firm-year panel
firm_year_panel <- funda_clean |>
  mutate(year = year(datadate)) |>
  left_join(firm_year_seg_agg, by = c("gvkey", "datadate")) |>
  mutate(
    # Firms with no segment record: treat as single-segment
    n_segments  = if_else(is.na(n_segments), 1L, as.integer(n_segments)),
    unique_sic2 = if_else(is.na(unique_sic2), 1L, as.integer(unique_sic2)),
    unique_sic3 = if_else(is.na(unique_sic3), 1L, as.integer(unique_sic3)),
    unique_kf49 = if_else(is.na(unique_kf49), 1L, as.integer(unique_kf49)),
    bh_div      = if_else(is.na(bh_div), 0, bh_div),
    is_conglomerate = unique_sic2 >= 2L
  )

# Within-firm log growth rates for total assets and sales (one period)
firm_year_panel <- firm_year_panel |>
  arrange(gvkey, fyear) |>
  group_by(gvkey) |>
  mutate(
    lag_at      = lag(at),
    lag_sale    = lag(sale),
    dlnat       = log(at)   - log(lag_at),
    dlnsale     = log(sale) - log(lag_sale),
    # consecutive-year check (avoid spurious "growth" across gaps)
    yr_gap      = fyear - lag(fyear),
    dlnat       = if_else(yr_gap == 1, dlnat, NA_real_),
    dlnsale     = if_else(yr_gap == 1, dlnsale, NA_real_)
  ) |>
  ungroup() |>
  select(-yr_gap)

# Rolling 5-year moments: mu, sigma2, g, variance drag D
# Right-aligned: at year t, use growth observations from t-4 through t
roll_moments <- function(x, k = 5, min_obs = 4) {
  n <- length(x)
  mu <- rep(NA_real_, n); s2 <- rep(NA_real_, n)
  for (i in seq_len(n)) {
    window <- x[max(1, i - k + 1):i]
    valid  <- window[!is.na(window)]
    if (length(valid) >= min_obs) {
      mu[i] <- mean(valid)
      s2[i] <- var(valid)
    }
  }
  list(mu = mu, s2 = s2)
}

firm_year_panel <- firm_year_panel |>
  arrange(gvkey, fyear) |>
  group_by(gvkey) |>
  mutate(
    .roll_at   = list(roll_moments(dlnat)),
    mu_at      = map_dbl(.roll_at, ~ .x$mu[length(.x$mu)]),  # placeholder
  ) |>
  ungroup() |>
  select(-.roll_at)

# More robust per-firm rolling computation (the placeholder above doesn't
# vectorize; here is the actual implementation that works row-wise):
firm_year_panel <- firm_year_panel |>
  arrange(gvkey, fyear) |>
  group_by(gvkey) |>
  mutate(
    mu_at_5y      = rollapply(dlnat,   width = 5, FUN = mean, by.column = FALSE,
                              align = "right", fill = NA, partial = TRUE,
                              na.rm = TRUE),
    sigma2_at_5y  = rollapply(dlnat,   width = 5, FUN = var,  by.column = FALSE,
                              align = "right", fill = NA, partial = TRUE,
                              na.rm = TRUE),
    g_at_5y       = mu_at_5y - sigma2_at_5y / 2,
    D_at_5y       = mu_at_5y - g_at_5y,                            # = sigma2/2
    mu_sale_5y    = rollapply(dlnsale, width = 5, FUN = mean, by.column = FALSE,
                              align = "right", fill = NA, partial = TRUE,
                              na.rm = TRUE),
    sigma2_sale_5y = rollapply(dlnsale, width = 5, FUN = var, by.column = FALSE,
                               align = "right", fill = NA, partial = TRUE,
                               na.rm = TRUE),
    g_sale_5y     = mu_sale_5y - sigma2_sale_5y / 2,
    D_sale_5y     = mu_sale_5y - g_sale_5y
  ) |>
  ungroup()

# Drop the placeholder columns from the false start above
firm_year_panel <- firm_year_panel |>
  select(-any_of(c("mu_at")))

saveRDS(firm_year_panel, file.path(PROC_DIR, "firm_year_panel.rds"))
cat(sprintf("  Firm-year panel: %s rows, %d unique firms\n",
            format(nrow(firm_year_panel), big.mark = ","),
            length(unique(firm_year_panel$gvkey))))


# ==============================================================================
# 11. Build analysis-ready segment-year panel
# ==============================================================================

cat("\nBuilding segment-year panel with derived variables...\n")

segment_year_panel <- seg_for_panel |>
  filter(!is.na(ias), ias > 0) |>
  arrange(gvkey, sid, datadate) |>
  group_by(gvkey, sid) |>
  mutate(
    lag_ias    = lag(ias),
    dlnias     = log(ias) - log(lag_ias),
    yr_gap     = year - lag(year),
    dlnias     = if_else(yr_gap == 1, dlnias, NA_real_),
    mu_ias_5y     = rollapply(dlnias, width = 5, FUN = mean, by.column = FALSE,
                              align = "right", fill = NA, partial = TRUE,
                              na.rm = TRUE),
    sigma2_ias_5y = rollapply(dlnias, width = 5, FUN = var, by.column = FALSE,
                              align = "right", fill = NA, partial = TRUE,
                              na.rm = TRUE),
    g_ias_5y      = mu_ias_5y - sigma2_ias_5y / 2,
    D_ias_5y      = mu_ias_5y - g_ias_5y
  ) |>
  ungroup() |>
  select(-yr_gap)

# Attach firm-level conglomerate status
segment_year_panel <- segment_year_panel |>
  left_join(
    firm_year_panel |> select(gvkey, datadate, unique_sic2, n_segments,
                              bh_div, is_conglomerate),
    by = c("gvkey", "datadate")
  )

saveRDS(segment_year_panel, file.path(PROC_DIR, "segment_year_panel.rds"))
cat(sprintf("  Segment-year panel: %s rows, %d unique segments\n",
            format(nrow(segment_year_panel), big.mark = ","),
            n_distinct(paste(segment_year_panel$gvkey,
                             segment_year_panel$sid))))


# ==============================================================================
# 12. Quick diagnostics
# ==============================================================================

cat("\n--- Diagnostics ---\n")

cat("\nFirm-year panel coverage by decade:\n")
firm_year_panel |>
  mutate(decade = (fyear %/% 10) * 10) |>
  count(decade) |>
  print(n = Inf)

cat("\nConglomerate share over time (SFAS 131 break in 1998):\n")
firm_year_panel |>
  group_by(fyear) |>
  summarise(
    n_firms        = n(),
    pct_conglom    = mean(is_conglomerate, na.rm = TRUE) * 100,
    avg_segments   = mean(n_segments, na.rm = TRUE)
  ) |>
  filter(fyear %in% c(1985, 1990, 1995, 1997, 1998, 2000, 2005, 2010, 2015, 2020)) |>
  print()

cat("\nVariance drag by conglomerate status (post-SFAS 131 sample):\n")
firm_year_panel |>
  filter(fyear >= 1998, !is.na(D_at_5y)) |>
  group_by(is_conglomerate) |>
  summarise(
    n            = n(),
    mean_mu      = mean(mu_at_5y,     na.rm = TRUE),
    mean_sigma2  = mean(sigma2_at_5y, na.rm = TRUE),
    mean_g       = mean(g_at_5y,      na.rm = TRUE),
    mean_D       = mean(D_at_5y,      na.rm = TRUE)
  ) |>
  print()

cat("\n--- Done ---\n")
cat(sprintf("Raw data:       %s\n", normalizePath(RAW_DIR)))
cat(sprintf("Processed data: %s\n", normalizePath(PROC_DIR)))
cat("\nNext steps: matching (Test 1), industry-pair panel (Test 2), regressions.\n")
