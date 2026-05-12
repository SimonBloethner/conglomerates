# Empirical Validation Section: Design Document (v2)

**Paper:** Conglomerate Mergers: Effects on Growth and Competition
**Scope:** ~10 pages, validation tone, appended to existing theory/ABM paper
**Data:** Compustat North America Fundamentals + Historical Segments, Ken French industry returns
**Identification target:** Moderate — descriptive plus tests that distinguish the mechanism from alternatives

---

## 1. What the empirical section is doing

The theoretical paper claims that internal pooling under multiplicative dynamics raises the time-average growth rate of constituent firms beyond what passive aggregation or shareholder-side diversification can deliver. The validation section's job is twofold:

1. Show the patterns the model predicts appear in the data — and crucially, that the **variance channel** rather than the mean channel is doing the work. A diversification effect that operates through higher $\mu$ is a different theory.
2. Demonstrate at least one empirical signature that the cooperative-diversification mechanism predicts but alternative theories (agency, market power, taxes, portfolio diversification) do not.

Three to four well-executed tests serve these goals better than ten weak ones. The structure below targets ~10 pages with Tests 2 and 3 carrying the burden of distinguishing the mechanism, and Tests 1 and 4 providing supporting evidence.

---

## 2. Four-test structure

### Test 1 (~3.5 pages): Variance drag and the dose–response in $N$

**Headline framing.** The model implies $g = \mu - \sigma^2/(2N)$ — variance drag is the active mechanism, and it scales with $1/N$. Both pieces are testable.

**Sample unit.** Segment-year observations for segments inside multi-segment firms; firm-year observations for single-segment firms used as standalone matches. Match each segment-year to single-segment firms with the same primary SIC2, similar log-size (within $\pm 30\%$), and same year. Compute over a 5-year forward rolling window:

- $\mu_{it} = \overline{\Delta \ln \text{ias}_{it}}$ (or $\Delta \ln \text{at}_{it}$ for the firm-level match)
- $\sigma_{it}^2 = \text{Var}(\Delta \ln \text{ias}_{it})$
- $g_{it} = \mu_{it} - \sigma_{it}^2/2$
- Variance drag $D_{it} = \mu_{it} - g_{it} \approx \sigma_{it}^2/2$

**Specification 1A — variance drag headline.**

$$D_{it} = \beta_0 + \beta_1 \cdot \text{Conglomerate}_{it} + X_{it}'\gamma + \alpha_j + \alpha_t + \varepsilon_{it}$$

Prediction: $\beta_1 < 0$. This is the cleanest one-line statement of the theory.

**Specification 1B — three-equation diagnostic.** Run the same regression with $\sigma_{it}^2$, $\mu_{it}$, and $g_{it}$ as outcomes separately. Predictions: $\beta_1^\sigma < 0$, $\beta_1^\mu \approx 0$, $\beta_1^g > 0$. The diagnostic value is critical — if $\beta_1^\mu > 0$, the apparent benefit comes from selection on quality rather than from variance reduction, and the variance-drag headline is misleading. The three-equation breakdown is what confirms which channel is operating.

**Specification 1C — continuous diversification depth.** Replace the dummy with the number of distinct 2-digit SIC segments $N_{it}$:

$$D_{it} = \beta_0 + \beta_1 \cdot N_{it} + X_{it}'\gamma + \alpha_j + \alpha_t + \varepsilon_{it}$$

Prediction: $\beta_1 < 0$, with magnitude consistent with $\sigma^2/(2N)$ — i.e., the marginal effect of moving from 1 to 2 segments should be larger than from 4 to 5.

Run a parallel specification using the **Berry-Herfindahl diversification index** as a robustness measure: $\text{BH}_{it} = 1 - \sum_s w_s^2$, where $w_s$ is segment $s$'s asset share. BH $\in [0, 1-1/N]$, increasing in both number of segments and how evenly assets are distributed across them.

**Specification 1D — non-monotonicity.** Add a quadratic term:

$$D_{it} = \beta_0 + \beta_1 N_{it} + \beta_2 N_{it}^2 + X_{it}'\gamma + \alpha_j + \alpha_t + \varepsilon_{it}$$

Prediction for variance drag: $\beta_1 < 0$ and $\beta_2 \geq 0$ (variance reduction has diminishing returns). For $g_{it}$: $\beta_1 > 0$ and $\beta_2 < 0$ (inverted-U as coordination costs eventually dominate). The inverted-U on $g$ is the bridge to the diversification discount/premium debate — direct empirical evidence that the relationship is non-monotonic, not uniformly positive or negative.

**Controls** ($X$): log size, leverage (debt/assets), R&D intensity (xrd/at), age, intangibles share. Industry × year fixed effects to absorb common shocks.

**Heterogeneity:** Split the sample by SG&A/sales as a coordination-cost proxy. Industries with low coordination costs should support larger pooling benefits before the inverted-U turns down.

### Test 2 (~3 pages): Cross-industry correlation and conglomerate formation

**Headline framing.** The cooperative-diversification mechanism makes a unique prediction that competing theories of conglomerate formation do not: conglomerates should preferentially span industry pairs with **low return correlations**. Under correlated processes, pooling delivers smaller variance reduction (Peters and Adamou 2022); under independent processes, the benefit is maximal. Agency theory, market-power theory, and tax theory all predict different sorting patterns and none predicts low-correlation sorting. This makes Test 2 the most discriminating test in the section.

**Industry classification.** Map each Compustat segment's primary SIC (`sics1`) to one of the Ken French 49 industries. The KF49 classification is the standard, and the granularity is appropriate — finer classifications (SIC4) have too few firms per group to estimate stable industry-level correlations.

**Industry correlation matrix.** Compute pairwise correlations of monthly returns across KF49 industries using **standalone firm portfolios only** — important because using all firms would let conglomerate membership endogenously affect the estimated correlations. The Ken French data files provide value-weighted industry returns built from CRSP, which is appropriate for this purpose. Use rolling 60-month windows updated annually; this gives 49 × 48 / 2 = 1,176 industry-pair-year observations of $\rho_{jk,t}$.

**Specification 2A — descriptive null comparison.** For each observed conglomerate $c$ at time $t$ with segments spanning industries $I_c$, compute the average pairwise correlation:

$$\bar\rho_c = \frac{2}{|I_c|(|I_c|-1)} \sum_{j,k \in I_c, j<k} \rho_{jk,t}$$

Build a null distribution by drawing 10,000 synthetic conglomerates per actual conglomerate, where each synthetic conglomerate has the same number of segments as the actual one but draws industries with probability proportional to industry size (number of firms or total assets — equal-probability sampling overweights small industries). Compare the actual mean $\bar\rho$ across observed conglomerates to the null. Prediction: actual mean is significantly below null.

**Specification 2B — industry-pair-year logit.**

$$\Pr(\text{Pair}(j,k) \in \text{some conglomerate at } t) = \Phi\big(\beta_0 + \beta_1 \rho_{jk,t} + \beta_2 \sigma_j + \beta_3 \sigma_k + X_{jk,t}'\gamma\big)$$

Unit: industry-pair-year (1,176 pairs × $T$ years). Dependent variable: indicator for whether at least one Compustat firm spans both industries through reported segments at time $t$. Controls $X$: log industry size of each, average growth rate, geographic overlap (fraction of firms headquartered in same Census region), 2-digit SIC distance (proxy for vertical relatedness).

Prediction: $\beta_1 < 0$.

**Caveat to acknowledge in the text.** The cross-section of multi-segment firms reflects survivorship — conglomerates that formed across high-correlation pairs and dissolved are not observed. This biases *toward* finding the predicted effect. The cleanest version uses merger events to identify formation rather than the cross-section, which requires deal-level data we may not have. Note this honestly as a limitation rather than papering over it.

### Test 3 (~2 pages): Crisis amplification

**Headline framing.** Under multiplicative dynamics, variance drag $\sigma^2/2$ becomes the binding constraint exactly when $\sigma$ is large. Pooling benefits should therefore widen in high-volatility periods. This connects directly to Kuppuswamy and Villalonga (2016), who document a reversal of the diversification discount during 2007–2009.

**Specification:**

$$y_{it} = \beta_1 \cdot \text{Cong}_{it} + \beta_2 \cdot \text{Cong}_{it} \times \text{HighVol}_t + \beta_3 \cdot \text{HighVol}_t + X_{it}'\gamma + \alpha_j + \varepsilon_{it}$$

Two `HighVol` proxies, run separately:
- (a) Annual mean VIX in top quartile across the sample period (post-1990 only)
- (b) Crisis dummies for 2001–2002, 2008–2009, 2020

Outcomes $y_{it}$: $\sigma_{it}^2$, $g_{it}$, and variance drag $D_{it}$.

Predictions:
- $\beta_2 < 0$ when $y = \sigma^2$ (conglomerate variance reduction is amplified in crises)
- $\beta_2 > 0$ when $y = g$ (conglomerate growth advantage widens in crises)
- $\beta_2 < 0$ when $y = D$ (variance drag is differentially compressed for conglomerates when overall volatility rises)

**Robustness:** Rerun with the **continuous** $N_{it}$ measure interacted with `HighVol` instead of the dummy. The dose-response should be steeper in high-volatility periods.

### Test 4 (~1.5 pages): Aggregate market structure

**Headline framing.** The simulation predicts that markets with greater conglomerate presence exhibit lower concentration and higher rank mobility (Figures 4 and 5 in the paper). Test these aggregate signatures directly.

**Sample unit.** Industry-year panels at the 3-digit SIC level (sufficient firms per cell while preserving variation). Drop industries with fewer than 10 firms per year.

For each industry-year:
- $\text{HHI}_{jt} = \sum_i (\text{at}_{ij,t} / \sum_i \text{at}_{ij,t})^2$ — capital-share Herfindahl
- $\text{CR4}_{jt}$, $\text{CR8}_{jt}$ — concentration ratios of top 4 and top 8 firms
- $\text{Gini}_{jt}$ — Gini coefficient of asset shares
- $\text{Mobility}_{jt}$ — within-industry standard deviation of 5-year asset rank changes
- $\text{ConglomerateShare}_{jt}$ — fraction of industry assets owned by firms with $N \geq 2$ distinct SIC2 segments

**Specification:**

$$y_{jt} = \beta_0 + \beta_1 \cdot \text{ConglomerateShare}_{jt} + \beta_2 \cdot \text{ConglomerateShare}_{jt}^2 + X_{jt}'\gamma + \alpha_j + \alpha_t + \varepsilon_{jt}$$

Predictions:
- For HHI, CR4, CR8, Gini: $\beta_1 < 0$ (concentration declines with conglomerate presence)
- For Mobility: $\beta_1 > 0$
- The quadratic term tests for non-monotonicity at the aggregate level — at very high conglomerate share, coordination-cost constraints could reverse the pattern

**Caveat:** This is the test most exposed to reverse causality. Frame as descriptive, document the partial correlations with industry and year fixed effects, do not claim causality. Lagged conglomerate share as a robustness check addresses simultaneity but not omitted-variable concerns.

---

## 3. Methodological issues to handle openly

These are the issues a referee will raise. Each is straightforward to address; acknowledging them strengthens rather than weakens the section.

### 3.1 The SFAS 131 regime change

Compustat segment data is not internally consistent across time. Pre-1998 (SFAS 14) allowed broad and often vague segment definitions. From 1998 onwards (SFAS 131) requires "operating segments" matching internal management structure. The reporting regime change increased reported segment counts for many firms by reporting fiat, not by changes in actual diversification.

**Implications:** Run the main analysis on the SFAS 131 era (1998–2024). Robustness on the full panel. For pre-1998, treat the diversification dummy as conservatively measured. Cite Berger and Hann (2003 RAS) and Botosan and Stanford (2005 TAR).

### 3.2 Segment data is noisy

Three specific issues:
1. **Segment IDs (`sid`) change over time.** If primary SIC changes, treat as a new segment.
2. **Geographic and business segments report separately.** Adding segment types double-counts. Restrict to `stype IN ('BUSSEG', 'OPSEG')`. Drop GEOSEG for the main analysis.
3. **Single-segment firms often have no segment record at all.** Treat firms in `comp.funda` with no match in `comp.wrds_segmerged` as single-segment, with spot-check verification.

### 3.3 Survivorship and the variance-reduction story

The single most likely way to get a Type II error. The model predicts conglomerates reduce extreme negative outcomes. If the sample conditions on firm survival, the standalone firms killed by extreme negative shocks are systematically dropped — exactly the observations driving the variance comparison. Include exiting firms with their full histories until exit. Treat financial delisting (`dlrsn` codes 02, 03) as a final $\delta_t \approx 0$ observation rather than missing data.

### 3.4 Selection: who diversifies?

The choice to be a conglomerate is endogenous. Industry × size × year fixed effects absorb cross-sectional sorting on observable dimensions. Firm fixed effects in a sub-sample of firms that transition between focused and diversified provide additional discipline. For a moderate-identification validation section, the right framing is that this is a correlational test consistent with the mechanism, not a causal test. State this explicitly.

### 3.5 The asset-vs-sales choice

The model is about capital accumulation, so total assets (`at`) is the correct primary measure. Sales (`sale`) is what most of the diversification-discount literature uses. Run main on `at`, robustness on `sale`. Note the alignment with the simulation's state variable in the text.

---

## 4. Exact Compustat data pulls (WRDS)

### Pull 1 — `comp.funda` (Fundamentals Annual)

**Variables:**

| Variable | Description | Use |
|---|---|---|
| `gvkey` | Firm ID | Linking |
| `conm` | Company name | Sanity checks |
| `fyear`, `datadate` | Year, fiscal year-end | Time dimension |
| `at` | Total assets | **Primary growth variable** |
| `sale` | Net sales | Robustness growth variable |
| `oibdp`, `ebit`, `ni` | Operating income, EBIT, net income | Profitability controls |
| `xsga` | SG&A expense | Coordination cost proxy |
| `xrd` | R&D expense | Intangibles control |
| `capx` | Capital expenditure | Investment control |
| `emp` | Employees | Size robustness |
| `dlc`, `dltt` | Short/long-term debt | Leverage |
| `ceq`, `prcc_f`, `csho` | Equity, price, shares | Market cap, B/M |
| `sich`, `naicsh` | Historical SIC, NAICS | Industry classification |
| `loc`, `fic`, `exchg` | Location, exchange | Sample filters |
| `dlrsn` | Delisting reason | Survivorship handling |

**WRDS Cloud SQL:**

```sql
SELECT gvkey, conm, fyear, datadate,
       at, sale, oibdp, ebit, ni, xsga, xrd, capx, emp,
       dlc, dltt, ceq, prcc_f, csho,
       sich, naicsh, loc, fic, exchg, dlrsn
FROM comp.funda
WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D'
  AND consol = 'C' AND curcd = 'USD'
  AND fyear BETWEEN 1980 AND 2024
  AND at IS NOT NULL AND at > 0;
```

**Sample exclusions** (apply post-pull, not in SQL):
- Drop SIC 6000–6999 (financials) for main sample; run separately as robustness
- Drop SIC 4900–4999 (regulated utilities)
- Drop SIC 9000–9999 (public administration)

Expected size: ~450,000 firm-year observations.

### Pull 2 — `comp.wrds_segmerged` (Historical Segments, merged)

**Variables:**

| Variable | Description |
|---|---|
| `gvkey`, `datadate`, `srcdate` | Linking and dedup keys |
| `stype` | Segment type — filter to BUSSEG, OPSEG |
| `sid`, `snms` | Segment ID and name |
| `sics1`, `sics2`, `naicss1` | Segment industry classification |
| `sales`, `ias`, `ops`, `emps`, `capxs` | Segment financials |

**Key derived variables to construct after the pull:**

- `n_segments_jt` — count of distinct `sid` per firm-year
- `unique_sic2_jt` — count of distinct 2-digit SIC across segments (the firmer measure)
- `bh_div_jt` — Berry-Herfindahl diversification index: $1 - \sum_s (\text{ias}_s / \sum_s \text{ias}_s)^2$
- `is_conglomerate_jt` — indicator for `unique_sic2_jt ≥ 2`
- `kf49_segments_jt` — set of Ken French 49 industries the firm spans

**WRDS Cloud SQL:**

```sql
SELECT gvkey, datadate, srcdate, stype, sid, snms,
       sics1, sics2, naicss1, sales, ias, ops, emps, capxs
FROM comp.wrds_segmerged
WHERE stype IN ('BUSSEG', 'OPSEG')
  AND datadate BETWEEN '1980-01-01' AND '2024-12-31'
  AND srcdate = datadate;
```

Expected size: ~600,000 segment-year observations.

### Pull 3 — Ken French data (free, central to Test 2)

From `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`:

1. **49 Industry Portfolios — value-weighted monthly returns** (1926–present). The standard correlation input.
2. **Industry classification file** — SIC code ranges mapped to KF49 industry numbers. Apply this mapping to each segment's `sics1`.

**Construction:**
- For each year $t$, compute the 49 × 49 correlation matrix using the trailing 60 months of returns
- Stack into industry-pair-year panel: 1,176 unordered pairs × $T$ years
- Merge with segment data via the SIC-to-KF49 mapping

### Pull 4 — Macro volatility (FRED, free)

For Test 3:
- VIX from `https://fred.stlouisfed.org/series/VIXCLS` (daily, 1990–present)
- VXO from `https://fred.stlouisfed.org/series/VXOCLS` (pre-VIX backward extension)

Aggregate to annual mean. Pre-1990: realized volatility of monthly S&P 500 returns from CRSP, or use the crisis-dummies approach only.

### Pull 5 (optional) — CRSP/Compustat link

Only needed if adding stock-return-based volatility as a robustness check.

```sql
SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt
FROM crsp.ccmxpf_lnkhist
WHERE linktype IN ('LC', 'LU')
  AND linkprim IN ('P', 'C');
```

### Pull 6 (deferred) — M&A deals

Skip for the validation section. If you confirm SDC/Refinitiv access, this opens a clean event-study extension for a follow-up paper that addresses the survivorship caveat in Test 2.

---

## 5. Suggested order of operations

**Week 1 — Data assembly**
1. Run Pulls 1 and 2. Save raw to a project directory.
2. Build the firm-year panel: merge `funda` with derived segment-level aggregates, flag single-segment vs multi-segment, count segments, compute Berry-Herfindahl.
3. Construct rolling 5-year growth-rate moments ($\mu$, $\sigma^2$, $g$, $D$) at firm and segment level.
4. Stylized-fact sanity check: confirm Stanley-style $\sigma$-size relationship and Laplace-shaped growth distribution (~half a page in the paper, but essential).

**Week 2 — Test 1**
5. Build standalone-firm matched sample. 1:3 nearest-neighbor matching on log size within SIC2 × year, 30% caliper.
6. Run Specifications 1A–1D. Iterate on sample construction until descriptive picture stabilizes.

**Week 3 — Test 2 (the centerpiece)**
7. Map Compustat segment SICs to KF49 industries.
8. Build rolling industry correlation matrices.
9. Construct industry-pair-year panel with conglomerate-spanning indicator.
10. Run descriptive null comparison and logit specification. **Pilot first on 1998–2010 manufacturing only** — the methodology is intricate enough that a pilot saves rework.

**Week 4 — Tests 3 and 4 plus writing**
11. Add VIX/crisis interactions for Test 3.
12. Industry-year aggregates and concentration/mobility regressions for Test 4.
13. Write the section. Allocate ~3.5 pages to Test 1, ~3 pages to Test 2, ~2 pages to Test 3, ~1.5 pages to Test 4.

**Total realistic timeline:** about a month if this is your only project, two if juggling. The matching for Test 1 and the industry-pair construction for Test 2 are the time sinks.

---

## 6. What this section does not claim

Worth being explicit in the paper text:

- It does not claim causal identification of pooling effects on firm growth.
- It does not directly observe internal capital transfers (these are not reliably measurable in segment data, and a test based on inferred transfers would conflate the pooling mechanism with Stein-style winner-picking — a confound the paper specifically argues against).
- It does not test welfare effects of conglomeration.
- It does not adjudicate the diversification discount/premium debate, but provides direct evidence on the non-monotonicity that organizes that debate.

What it does establish: the variance-drag patterns predicted by the model are visible in the data, the variance channel rather than the mean channel is doing the work, conglomerates preferentially form across low-correlation industry pairs (a signature unique to the cooperative-diversification mechanism), and the effect amplifies in high-volatility periods. For a theoretical paper with an ABM, that is the appropriate empirical contribution.

---

## 7. Things to verify before starting

1. WRDS subscription includes `comp.wrds_segmerged` specifically (not always bundled with basic Compustat North America)
2. Ken French 49-industry SIC mapping covers the full sample period (1980–2024)
3. Co-author alignment on adding ~10 pages to a paper currently at ~33 pages
4. Whether to confirm SDC/Refinitiv access before committing to skip the event-study extension entirely
