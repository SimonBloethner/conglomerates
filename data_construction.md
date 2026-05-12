# Data Construction Notes

**Project:** Conglomerate Mergers — empirical validation section
**Last updated:** May 2026
**Author:** Simon Blöthner

This document records all decisions made during data acquisition and panel construction for the empirical section. It serves two purposes: (1) reference when writing the paper's data section, and (2) replication documentation for the final submission.

---

## 1. Data sources

The empirical section combines four data sources.

**Primary:** Compustat North America via Wharton Research Data Services (WRDS).

- **Fundamentals Annual** (`comp.funda`): firm-year accounting and financial data, 1950–present (used 1980–2024).
- **Historical Segments** (`comp_segments_hist_daily.wrds_segmerged`): segment-year reporting going back to 1976. This is a separately-licensed add-on to the basic Compustat North America subscription. See replication notes (§9) for the schema technicality.
- **Company header** (`comp.company`): static firm-level identifiers (country, state, delisting reason, IPO date).

**Auxiliary:**

- **Ken French 49 Industry Portfolios** (Dartmouth Tuck): value-weighted monthly returns and SIC-to-industry mapping. Used to construct industry-level return correlation matrices for Test 2 (cross-industry correlation and conglomerate formation).
- **VIX index** (FRED series `VIXCLS`): daily closing values 1990–present, aggregated to annual mean. Used as the high-volatility indicator for Test 3 (crisis amplification).

All data acquired via R using the `RPostgres` package against the WRDS PostgreSQL server (`wrds-pgdata.wharton.upenn.edu:9737`) and direct HTTP downloads for Ken French and FRED.

---

## 2. Sample period and standard filters

### Sample period
Fiscal years 1980 through 2024. The 1980 start avoids the sparse early Compustat coverage and pre-dates major reporting regime changes. Segment data starts in 1976 in the source, but very few firms report segments before 1980.

### Compustat standard filters
The following filters are applied to `comp.funda`. They are the conventions used throughout the empirical accounting and finance literature (see e.g. Berger and Ofek 1995; Fama and French 1992; the Tidy Finance reference).

| Filter | Value | Rationale |
|---|---|---|
| `indfmt` | `'INDL'` | Industrial accounting format only; excludes financial services format. |
| `datafmt` | `'STD'` | Standardized data; excludes restatements and non-standard formats. |
| `consol` | `'C'` | Consolidated reporting only; excludes non-consolidated subsidiaries. |
| `popsrc` | `'D'` | Domestic (North America) data source. |
| `curcd` | `'USD'` | US-dollar denominated reports only. |
| `at > 0` | non-null, positive | Total assets must be observed and positive. |

### Industry exclusions

Three industry groups are excluded from the main analysis:

- **Financials (SIC 6000–6999):** different accounting standards and capital structures; standard practice in non-financial diversification studies.
- **Regulated utilities (SIC 4900–4999):** regulated returns make growth and volatility non-comparable.
- **Public administration (SIC 9000–9999):** not for-profit entities subject to different reporting.

Financials are run separately as a robustness check.

---

## 3. Variable definitions

### Firm-level (from `comp.funda`)

| Variable | Compustat code | Use |
|---|---|---|
| Total assets | `at` | Primary capital measure; growth variable. |
| Net sales | `sale` | Robustness growth variable. |
| Operating income before D&A | `oibdp` | Profitability control. |
| Earnings before interest and tax | `ebit` | Profitability control. |
| Net income | `ni` | Profitability control. |
| SG&A expense | `xsga` | Coordination cost proxy. |
| R&D expense | `xrd` | Intangibles control. |
| Capital expenditure | `capx` | Investment control. |
| Employees | `emp` | Size robustness. |
| Short-term debt | `dlc` | Leverage. |
| Long-term debt | `dltt` | Leverage. |
| Common equity | `ceq` | Equity. |
| Closing price (fiscal year) | `prcc_f` | Market valuation. |
| Common shares outstanding | `csho` | Shares outstanding. |
| Historical SIC | `sich` | **Time-varying industry classification.** Used in preference to current SIC. |
| Historical NAICS | `naicsh` | Time-varying NAICS. |
| Stock exchange | `exchg` | Listing venue. |
| Foreign incorporation country | `fic` | Geographic classification (in funda). |

### Static firm-header (from `comp.company`)

| Variable | Description |
|---|---|
| `loc` | Current country of headquarters. Note: this is the current value; historical relocations are not preserved. |
| `state` | Current US state. Same caveat. |
| `dlrsn` | Research company deletion reason. String values: `"01"` = M&A, `"02"` = bankruptcy, `"03"` = liquidation, etc. |
| `dldte` | Deletion date (when firm dropped from Compustat). |
| `ipodate` | IPO date. |
| `sic`, `naics` | Current SIC/NAICS (renamed `sic_curr`, `naics_curr` to avoid confusion with the time-varying versions in funda). |

### Segment-level (from `comp_segments_hist_daily.wrds_segmerged`)

| Variable | Description |
|---|---|
| `sid` | Segment identifier within a firm-year. |
| `snms` | Segment name (text). |
| `stype` | Segment type. Restricted to `'BUSSEG'` (business segment) and `'OPSEG'` (operating segment). |
| `sics1`, `sics2` | Primary and secondary SIC codes for the segment. |
| `naicss1` | Primary NAICS for the segment. |
| `sales` | Segment revenue. |
| `ias` | **Identifiable assets** — segment-level capital measure; primary growth variable for segment-level tests. |
| `ops` | Segment operating profit. |
| `emps`, `capxs` | Segment employees and capex. |
| `datadate` | Fiscal year-end date the data pertains to. |
| `srcdate` | Source date — when the record was created or revised. |

### Derived variables (constructed in R)

For each firm-year, we construct:

- `n_segments` — count of distinct segment IDs reported.
- `unique_sic2` — count of distinct 2-digit SIC codes across the firm's segments. The primary diversification indicator.
- `unique_sic3` — robustness at 3-digit level.
- `unique_kf49` — distinct Ken French 49-industry classifications.
- `bh_div` — Berry-Herfindahl diversification index: $1 - \sum_s w_s^2$ where $w_s$ is segment $s$'s share of total identifiable assets. Continuous measure ranging from 0 (single segment) to $1 - 1/N$ (perfectly diversified).
- `is_conglomerate` — indicator for `unique_sic2 ≥ 2`. **A firm is classified as a conglomerate if it reports two or more business segments in different 2-digit SIC codes.** Firms not appearing in `wrds_segmerged` are treated as single-segment (verified ~95% segment-data coverage; see §6).

For growth rate computation:

- `dlnat` — log change in total assets from year $t-1$ to $t$. Computed only when consecutive fiscal years are observed (gap of exactly 1 year); non-consecutive observations get `NA` to avoid spurious "growth" rates across data gaps.
- `dlnsale` — log change in net sales, same treatment.

For the rolling moments (5-year right-aligned windows, minimum 4 valid observations):

- $\mu_{it}$ — mean of `dlnat` over the window.
- $\sigma^2_{it}$ — variance of `dlnat` over the window.
- $g_{it} = \mu_{it} - \sigma^2_{it}/2$ — time-average growth rate; the model's primary object of interest.
- $D_{it} = \mu_{it} - g_{it} \approx \sigma^2_{it}/2$ — variance drag.

---

## 4. Conglomerate definition

A firm-year is classified as a conglomerate when:
- It reports two or more business segments (`BUSSEG` or `OPSEG`) in `comp_segments_hist_daily.wrds_segmerged`, AND
- These segments span two or more distinct 2-digit SIC codes.

Geographic segments (`GEOSEG`) are excluded to avoid double-counting (geographic and business segments both report at the firm-year level and adding them sums to roughly twice the firm's totals). Customer segments are excluded as not relevant to production-side diversification.

The 2-digit SIC criterion is the standard convention in the diversification literature; using only segment count (without SIC distinction) would incorrectly classify firms with multiple segments in related industries as diversified.

---

## 5. Survivorship and delisting handling

The model predicts conglomerates reduce extreme negative outcomes. If the panel conditions on firm survival, this systematically drops standalone firms killed by extreme negative shocks — exactly the observations driving the variance comparison. The sample therefore **retains exiting firms with their full histories up to the exit year.**

Delisting reasons from `comp.company.dlrsn`:

- Codes `"02"` (bankruptcy) and `"03"` (liquidation) represent failed firms — these exits are economically equivalent to a final $\delta_t \approx 0$ observation and are kept in the panel.
- Code `"01"` (acquired/merged) represents non-failures and is treated as the firm being absorbed rather than failing.
- Codes `"05"`, `"07"`, `"10"` (no longer files / other / other reasons) are treated as exits but without the bankruptcy interpretation.

This handling is essential to avoid Type II error: with strict survival conditioning, the variance-reduction prediction systematically loses statistical power because the high-variance standalone firms drop out of the sample.

---

## 6. Sample size and composition

After all filters, the analysis-ready panel contains:

| Panel | Observations | Unique units |
|---|---|---|
| Firm-year (`firm_year_panel`) | 225,612 | 21,462 firms |
| Segment-year (`segment_year_panel`) | 584,876 | 86,913 segments |

Source pull row counts:
- `comp.funda` after filters: 349,773 firm-year observations.
- `comp.company`: 57,479 firm header records.
- `comp_segments_hist_daily.wrds_segmerged`: 1,404,485 raw segment-year rows, 703,757 after deduplication.

### Coverage by decade

| Decade | Firm-years | Distinct firms | Segment coverage |
|---|---|---|---|
| 1980s | 17,652 | 11,575 | 97.6% |
| 1990s | 70,754 | 15,806 | 91.6% |
| 2000s | 61,635 | 13,537 | 96.4% |
| 2010s | 49,861 | 11,322 | 97.2% |
| 2020s | 25,710 | 8,277 | 97.0% |

"Segment coverage" is the share of firms in funda that have at least one record in `wrds_segmerged`. The remaining 3-8% are treated as single-segment, which is the standard assumption (verified by spot-checks of 10-K filings).

### Conglomerate share over time

| Year | Firms | % Conglomerate | Mean segments |
|---|---|---|---|
| 1990 | 5,754 | 20.5% | 1.47 |
| 1995 | 7,738 | 15.1% | 1.35 |
| 1997 | 7,788 | 14.3% | 1.37 |
| **1998** | **7,985** | **19.2%** | **1.66** |
| 2000 | 7,630 | 20.8% | 1.96 |
| 2005 | 5,982 | 19.1% | 2.04 |
| 2010 | 5,075 | 18.1% | 2.04 |
| 2020 | 5,055 | 12.9% | 1.70 |

Two patterns are visible:

**SFAS 131 break in 1998.** Mean segment count jumps from 1.37 (1997) to 1.66 (1998) and the conglomerate share rises from 14.3% to 19.2%. This is consistent with SFAS 131 requiring more disaggregated reporting based on internal management structure (see §7).

**Long-run conglomerate decline.** Conglomerate share peaks at ~21% in 2000 and falls to ~13% by 2020. This tracks the well-documented refocusing trend in US public firms over the 2000s and 2010s, driven by spinoffs and divestitures (Berger and Ofek 1999; Maksimovic and Phillips 2002).

### US public-firm decline

The total number of distinct firms drops from ~15,800 in the 1990s to ~8,300 in the 2020s — about a 47% decline. This is the well-documented US public-firm decline described by Doidge, Karolyi and Stulz (2017, JFE). The cross-sectional tests are unaffected; time-series interpretation should note that sample composition changes meaningfully.

---

## 7. The SFAS 131 regime change

In 1998, FASB Statement No. 131 replaced SFAS 14 as the segment-reporting standard. The change matters substantially for diversification research:

- **Pre-1998 (SFAS 14):** allowed broad and often vague segment definitions. Firms frequently reported as single-segment when economically diversified.
- **Post-1998 (SFAS 131):** requires reporting of "operating segments" matching the internal management structure used by the chief operating decision maker. This is a tighter, more disclosure-intensive standard.

Empirical implications, supported by Berger and Hann (2003 RAS) and Botosan and Stanford (2005 TAR):

- Reported segment counts rose by reporting fiat, not by changes in actual diversification.
- The conglomerate indicator is more conservative (likely understated) pre-1998.
- Variance-reduction predictions should hold in both eras if the mechanism is real, but the cross-sectional variation in conglomerate status is larger and cleaner post-1998.

**Main analysis sample: 1998–2024.** Robustness on the full panel including 1980–1997.

---

## 8. Auxiliary data sources

### Ken French 49 Industry Portfolios

Downloaded via the `frenchdata` R package from Dartmouth Tuck's data library. Source URL: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`.

- Value-weighted monthly returns for 49 industry portfolios, July 1926–present (58,653 industry-month observations within the sample period).
- Constructed by Fama-French from CRSP firms classified into 49 industries via SIC ranges.

Used in two ways:
- Industry-level return correlation matrices for Test 2: rolling 60-month windows, updated annually, yielding 1,176 unordered industry pairs $\times$ years.
- Mapping of each Compustat segment's `sics1` to one of the 49 industries via the SIC-range classification file (`Siccodes49.zip` from the same source).

SIC-to-KF49 mapping: 5,054 unique 4-digit SIC codes mapped. Unmapped codes default to industry 49 ("Other").

**Important:** correlations are estimated from value-weighted industry portfolios built from the universe of firms — including conglomerates' constituent segments. There is a small endogeneity concern (conglomerate membership could affect estimated industry correlations), but it is second-order because no single conglomerate dominates an industry's value-weighted portfolio. A robustness check restricting to standalone firms only could be added if a referee raises this.

### VIX index

Source: Federal Reserve Bank of St. Louis (FRED), series `VIXCLS`. Direct CSV download from `https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS`.

- Daily closing values, January 1990–present (9,181 daily observations).
- Aggregated to annual mean (37 annual observations 1990–partial 2026).
- For pre-1990 sample years, no VIX value is available. Test 3 (crisis amplification) is run on 1990 onward, with crisis-period dummies as an alternative specification for the pre-1990 robustness check.

---

## 9. Replication notes (technical)

These details matter for replication but probably not for the paper text.

### Compustat schema convention

A subtle but project-critical point: WRDS hosts Compustat segment data in **two different schemas** with different coverage:

- **`comp.wrds_segmerged`** contains only the **rolling 7-year** segment data ("Segments - Non-Historical"). This is part of the basic Compustat North America subscription.
- **`comp_segments_hist_daily.wrds_segmerged`** contains the **full historical** segment data back to 1976. This is the "Historical Segments" product, a separately-licensed add-on.

Both tables have identical column structure. A researcher querying `comp.wrds_segmerged` without realizing the schema distinction will get truncated data and may falsely conclude their subscription is limited. Confirm via:

```sql
SELECT MIN(datadate), MAX(datadate), COUNT(*)
FROM comp_segments_hist_daily.wrds_segmerged;
```

### `comp.funda` vs `comp.company`

Static firm-header variables (`loc`, `fic`, `state`, `dlrsn`, `dldte`, `ipodate`, current `sic`/`naics`) live in `comp.company`, not `comp.funda`. They must be pulled separately and joined on `gvkey`. The funda table does have `fic` (foreign incorporation country) — this is one variable that appears in both tables.

The company table stores **current** values only. Historical state/country/SIC changes are not preserved here. For state-level identification studies, historical headquarters data from SEC filings (Bai, Fairhurst, Serfling 2020 RFS) is required.

### Srcdate deduplication

The segment merged table can theoretically contain multiple records per `(gvkey, datadate, sid)` due to data restatements (`srcdate > datadate`). The convention is to keep the earliest `srcdate` (as-first-reported).

In our pull this dedup turned out to be irrelevant — each `(gvkey, datadate, sid, stype)` combination appeared only once in the data. But the dedup logic is included in the script for portability to other vintages or other WRDS instances where multiple records might appear.

### Industry classification choice

`sich` (historical SIC from funda) is used in preference to `sic` (current SIC from company). The historical SIC reflects the firm's primary industry at the time of each fiscal year, allowing for industry reclassifications. Using the current SIC would treat industry membership as time-invariant, which it is not.

### Auxiliary R packages used

- `DBI`, `RPostgres` — WRDS connection
- `dplyr`, `tidyr`, `purrr`, `readr`, `stringr`, `lubridate` — data manipulation
- `zoo` — rolling-window computations
- `frenchdata` — Ken French data download

Version pinning will be added at the analysis stage when the replication package is finalized.

---

## 10. Known limitations and caveats

These should be acknowledged in the paper's data section or limitations subsection.

**Selection on diversification status.** The choice to be a conglomerate is endogenous. Firms may diversify for reasons correlated with their growth trajectory, biasing simple comparisons. Industry-year fixed effects absorb cross-sectional sorting on observable industries, and matched-sample designs further address this, but the validation section does not claim full causal identification.

**Segment definition discretion.** Firms have substantial latitude in how they aggregate operations into reported segments. SFAS 131 reduced this discretion but did not eliminate it. The Berger-Ofek (1995) versus Villalonga (2004) debate is in part about this.

**Survivorship in segment tracking.** Segments may be redefined, consolidated, or split over time. We treat each unique `(gvkey, sid)` pair as a segment, but if a firm reorganizes its segment structure, this may show up as segment exit and new segment entry. For variance-reduction tests this likely biases against finding the effect (more apparent variability in segment-level series), so it is conservative.

**Industry correlation estimates.** Built from Ken French value-weighted portfolios that include conglomerates' constituent assets. Strict identification would estimate correlations from standalone firms only. This is a refinement worth doing as robustness.

**No deal-level M&A data.** Without SDC Platinum or Refinitiv data, the cross-sectional conglomerate variable cannot be cleanly traced to specific merger events. The cross-sectional tests are interpretable as "firms with this status today" rather than "firms that experienced this merger." An event-study extension using deal-level data could be a follow-up paper.

**The 2017+ vs 1976+ subscription distinction (note for replication).** Researchers attempting to replicate this work need access to the Historical Segments add-on at their WRDS-subscribing institution. The basic Compustat North America subscription does not include it.

---

## 11. File outputs

After running `01_acquire_data.R`, the following files exist in `data/`:

**`raw/`** (untransformed pulls):
- `funda_raw.rds` — Compustat Fundamentals merged with company header
- `segments_raw.rds` — Compustat Historical Segments after srcdate dedup
- `kf49_returns.rds` — Ken French 49-industry monthly returns
- `kf49_sicmap.rds` — SIC-to-KF49 lookup table
- `kf49_ranges.rds` — Original KF49 SIC range specifications
- `vix_daily.rds` — Daily VIX values
- `vix_annual.rds` — Annual VIX summaries

**`processed/`** (analysis-ready):
- `firm_year_panel.rds` — 225,612 firm-years, 29 base variables + derived diversification measures + rolling growth moments
- `segment_year_panel.rds` — 584,876 segment-years with segment-level rolling moments and parent-firm conglomerate status

These files form the basis of the analysis scripts (`02_test1_matching_and_regressions.R` and subsequent).
