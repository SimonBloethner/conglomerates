# C41: growth-rate distributions of members and standalones

Diagnostics on the 27 pilot D traces (moment rule, main block, power law; normal, Laplace, t₃ × α ∈ {0.05, 0.1, 0.3}
× reps 0–2). No model change, no new model runs. Script `analysis/growth_distributions.py <trace_dir> --pilot d`
(`--tables` prints the tables below from the CSVs); output `pilot_d/growth_distributions.csv` (900 rows: tables
`status`, `K`, `decile` × four measures), `pilot_d/growth_distributions_exclusions.csv`,
`figures/appendix/growth_density_alpha0.3.{pdf,png}`. Departures from the brief: `diagnostics/deviations.md`, C41.

## Answer

(a) **Shape.** Members' growth rates under Laplace shocks stay tent-shaped: on all firm-periods (pre-lift,
standardized by qₘ) their MAD/SD is 0.714–0.722 (Laplace 0.707; standalones 0.713–0.714), the excess kurtosis
2.81–2.83 (standalones 2.84–2.87), and the ML Laplace fit beats the normal by 0.053–0.063 nats per observation
(standalones 0.063–0.064). The mixture does not pull the normal family towards Laplace in any way that matters:
normal members keep MAD/SD 0.796–0.798 (normal 0.798) and excess kurtosis 0.00–0.07, and the normal fit stays
ahead by 0.046–0.049 nats. It does pull t₃ towards Laplace: members' MAD/SD rises from the standalones' 0.606–0.623
to 0.643–0.646 and the Laplace-over-normal margin falls from 0.20–0.23 to 0.16–0.17 nats; they remain heavier-tailed
than Laplace. (t₃ sample kurtosis is set by a handful of extreme draws, 240–1,270, and is not informative.) Pooling
mixes shocks, but a member's own shock keeps a weight of at least 1 − α, so the family survives.

(b) **Scale.** At α = 0.3 the members' growth sd is 0.773 of the standalones' under both normal and Laplace
shocks: 0.870 at K = 2, 0.809 at K = 3–4 and 0.755 at K = 5+ (mean K 8.2) under Laplace. The equal-members
reference √(1 − α(2 − α)(1 − 1/K)) gives 0.863, 0.796 and 0.743 at the bins' mean K; it is a little low because
units are unequal. With the units' actual capital weights, √mean[(1 − α + α wᵢ)² + α²(Σ w² − wᵢ²)], the reference is
0.867, 0.806 and 0.756, within 0.005 of the observed ratio in every bin for normal and Laplace. So the scale of a
member's growth is what linear pooling predicts. Under t₃ the sd ratio (0.81, 0.76, 0.72) falls below the
reference because the sd is driven by rare extremes, which pooling dilutes; the IQR ratio is 0.80 (reference
0.78). At α = 0.05 and 0.1 the fall is 3–8% for normal and Laplace, again matching the reference.

(c) **Tails.** The lower tail is cut more than the scale, the upper tail less. At α = 0.3 under Laplace the members'
1st percentile is 0.714 of the standalones' (−2.00 against −2.81) while their sd is 0.773; the 99th percentile is
0.819 (+2.24 against +2.74). So the lower tail is compressed 8% beyond the scale fall, and members' growth is
right-skewed (+0.28; standalones +0.01). Normal: p1 ratio 0.749, p99 0.805, sd 0.773. t₃: p1 0.699, p99 0.861, sd
0.740. The asymmetry is the pooling equation: a member's gross growth is the arithmetic mix (1 − α) e^{xᵢ} + α Σ wⱼ
e^{xⱼ} − Φ, so a large own loss is floored by the partners' share while a large own gain passes through, shifted
by log(1 − α). In the figure (Laplace panel) members' lower tail bends below the fitted Laplace from about −4 qₘ
while the upper tail stays on it. This is the cut in the lower tail of members' growth that Section 4.6 attributes
the Hill result to.

**Size and volatility.** Standalones show no size–volatility relation: on all firm-periods the sd by decile of
log share is flat (D10/D1 1.000–1.005 for normal and Laplace, 0.996–1.039 for the noisier t₃). Members' sd rises
with size (D10/D1 1.03–1.10, largest at α = 0.3, concentrated in the top two deciles), and the actual-weights
reference predicts it (1.02–1.11): a large member is a large part of its own unit and pools less of its own shock.
So the model has no Stanley-type falling size–volatility relation by construction, and pooling gives, if
anything, the opposite slope. The brief's measures show a spurious slope for standalones (D10/D1 1.24–1.46): the
exclusion rule drops exactly the periods in which a firm near the floor draws a large loss, which narrows small
firms' distribution, and in raw share growth the market term also lowers the top decile.

## Measures and sample

- **Growth over step t** (row t → t + 1 of the trace, steps 8,000–10,999). Within a step the order is mergers,
  pooling, floor, exits, so a firm is a member for that step if it was pooled: in a unit at the start of the step or
  at its end. K is the number of firms pooled in that unit in that step.
- **raw**: Δ log s, the change in log market share (as the brief asks). Pilot D fixes each market's mean size, so
  this is the firm's log growth minus the log growth of its market's mean size.
- **demeaned**: raw minus the market-period mean over the non-excluded firms; the common market term cancels.
- **standardized**: demeaned / qₘ (the market's shock IQR). Markets differ in qₘ (0.1–0.3), so without this even
  standalones pool into a scale mixture fatter-tailed than their family.
- **prelift**: all firm-periods, lifts included, with the growth before the lift (raw minus the trace's floor
  jump), demeaned over all the market's firms, / qₘ. The answer above uses this measure.
- The brief's exclusion (a lift in step t or t − 1) removes 8.8–17.9% of firm-periods, nine-tenths of them
  standalones:

| Family | α | firm-periods | lift in step | lift in previous step only | excluded % | of which members | kept members | kept standalones |
|---|---|---|---|---|---|---|---|---|
| laplace | 0.05 | 22,500,000 | 2,468,138 | 1,134,119 | 16.01 | 376,087 | 12,514,286 | 6,383,457 |
| laplace | 0.1 | 22,500,000 | 2,359,536 | 1,083,552 | 15.30 | 364,951 | 13,192,374 | 5,864,538 |
| laplace | 0.3 | 22,500,000 | 1,879,904 | 861,821 | 12.19 | 288,989 | 15,152,660 | 4,605,615 |
| normal | 0.05 | 22,500,000 | 1,863,183 | 882,954 | 12.21 | 287,028 | 13,869,402 | 5,884,461 |
| normal | 0.1 | 22,500,000 | 1,763,099 | 835,725 | 11.55 | 276,341 | 14,636,870 | 5,264,306 |
| normal | 0.3 | 22,500,000 | 1,339,149 | 635,970 | 8.78 | 207,734 | 16,540,388 | 3,984,493 |
| t3 | 0.05 | 22,500,000 | 2,767,216 | 1,269,911 | 17.94 | 421,524 | 11,666,781 | 6,796,092 |
| t3 | 0.1 | 22,500,000 | 2,665,424 | 1,222,866 | 17.28 | 409,959 | 12,305,711 | 6,305,999 |
| t3 | 0.3 | 22,500,000 | 2,229,546 | 1,021,685 | 14.45 | 338,406 | 14,094,327 | 5,154,442 |

## By status, prelift (all firm-periods, standardized)

Normal with IQR 1: sd 0.741, MAD/SD 0.798. Laplace with IQR 1: sd 1.020, MAD/SD 0.707, excess kurtosis 3. The
standalones match (demeaning over the market's ~50 firms removes 2% of the variance).

| Family | α | status | n | mean | sd | IQR | skew | ex. kurt | p1 | p5 | p95 | p99 | MAD/SD | LL Laplace | LL normal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | member | 14,156,430 | +0.0047 | 0.7084 | 0.9560 | +0.01 | 0.00 | -1.638 | -1.158 | +1.172 | +1.659 | 0.798 | -1.123 | -1.074 |
| normal | 0.05 | standalone | 8,343,570 | -0.0079 | 0.7339 | 0.9897 | +0.01 | -0.00 | -1.711 | -1.213 | +1.202 | +1.703 | 0.798 | -1.158 | -1.110 |
| normal | 0.1 | member | 14,913,211 | +0.0055 | 0.6809 | 0.9182 | +0.03 | 0.01 | -1.566 | -1.109 | +1.130 | +1.604 | 0.798 | -1.083 | -1.035 |
| normal | 0.1 | standalone | 7,586,789 | -0.0109 | 0.7334 | 0.9896 | +0.01 | -0.00 | -1.713 | -1.215 | +1.198 | +1.698 | 0.798 | -1.157 | -1.109 |
| normal | 0.3 | member | 16,748,122 | +0.0071 | 0.5668 | 0.7604 | +0.07 | 0.07 | -1.290 | -0.912 | +0.950 | +1.361 | 0.796 | -0.897 | -0.851 |
| normal | 0.3 | standalone | 5,751,878 | -0.0206 | 0.7335 | 0.9889 | +0.01 | 0.00 | -1.723 | -1.225 | +1.189 | +1.690 | 0.798 | -1.157 | -1.109 |
| laplace | 0.05 | member | 12,890,373 | +0.0097 | 0.9729 | 0.9743 | +0.06 | 2.83 | -2.642 | -1.566 | +1.600 | +2.712 | 0.714 | -1.328 | -1.391 |
| laplace | 0.05 | standalone | 9,609,627 | -0.0131 | 1.0104 | 1.0106 | -0.00 | 2.87 | -2.796 | -1.658 | +1.628 | +2.766 | 0.713 | -1.365 | -1.429 |
| laplace | 0.1 | member | 13,557,325 | +0.0117 | 0.9344 | 0.9392 | +0.11 | 2.81 | -2.511 | -1.497 | +1.545 | +2.625 | 0.715 | -1.289 | -1.351 |
| laplace | 0.1 | standalone | 8,942,675 | -0.0178 | 1.0099 | 1.0107 | -0.00 | 2.87 | -2.799 | -1.661 | +1.623 | +2.759 | 0.713 | -1.365 | -1.429 |
| laplace | 0.3 | member | 15,441,649 | +0.0162 | 0.7801 | 0.8067 | +0.28 | 2.82 | -2.004 | -1.221 | +1.308 | +2.243 | 0.721 | -1.118 | -1.171 |
| laplace | 0.3 | standalone | 7,058,351 | -0.0354 | 1.0087 | 1.0132 | +0.01 | 2.84 | -2.808 | -1.674 | +1.605 | +2.740 | 0.714 | -1.365 | -1.428 |
| t3 | 0.05 | member | 12,088,305 | +0.0152 | 1.0810 | 0.9772 | +3.48 | 241.53 | -2.764 | -1.455 | +1.497 | +2.883 | 0.643 | -1.330 | -1.497 |
| t3 | 0.05 | standalone | 10,411,695 | -0.0177 | 1.1618 | 1.0125 | -6.03 | 759.82 | -2.980 | -1.553 | +1.514 | +2.899 | 0.623 | -1.370 | -1.569 |
| t3 | 0.1 | member | 12,715,670 | +0.0180 | 1.0429 | 0.9437 | +4.20 | 279.05 | -2.630 | -1.396 | +1.450 | +2.806 | 0.643 | -1.294 | -1.461 |
| t3 | 0.1 | standalone | 9,784,330 | -0.0234 | 1.1672 | 1.0122 | -6.40 | 790.06 | -2.988 | -1.558 | +1.508 | +2.892 | 0.620 | -1.370 | -1.574 |
| t3 | 0.3 | member | 14,432,733 | +0.0248 | 0.8881 | 0.8149 | +5.01 | 295.43 | -2.116 | -1.168 | +1.258 | +2.474 | 0.646 | -1.137 | -1.300 |
| t3 | 0.3 | standalone | 8,067,267 | -0.0443 | 1.2003 | 1.0162 | -10.38 | 1271.51 | -3.026 | -1.583 | +1.492 | +2.873 | 0.606 | -1.375 | -1.602 |

## By status, standardized (the brief's sample: lift periods excluded)

Same pattern for shape and scale. The standalones' lower tail is truncated by the exclusion (Laplace p1 −2.27 to
−2.30 against the shock's −2.82), so tail comparisons use the table above.

| Family | α | status | n | mean | sd | IQR | skew | ex. kurt | p1 | p5 | p95 | p99 | MAD/SD | LL Laplace | LL normal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | member | 13,869,402 | -0.0184 | 0.7007 | 0.9437 | +0.02 | 0.01 | -1.642 | -1.167 | +1.138 | +1.623 | 0.797 | -1.111 | -1.063 |
| normal | 0.05 | standalone | 5,884,461 | +0.0433 | 0.6985 | 0.9298 | +0.02 | 0.08 | -1.589 | -1.102 | +1.199 | +1.692 | 0.794 | -1.104 | -1.060 |
| normal | 0.1 | member | 14,636,870 | -0.0170 | 0.6741 | 0.9074 | +0.03 | 0.02 | -1.573 | -1.119 | +1.098 | +1.570 | 0.797 | -1.072 | -1.025 |
| normal | 0.1 | standalone | 5,264,306 | +0.0474 | 0.6962 | 0.9266 | +0.02 | 0.08 | -1.580 | -1.094 | +1.200 | +1.691 | 0.794 | -1.100 | -1.057 |
| normal | 0.3 | member | 16,540,388 | -0.0120 | 0.5627 | 0.7540 | +0.07 | 0.07 | -1.299 | -0.923 | +0.925 | +1.335 | 0.796 | -0.890 | -0.844 |
| normal | 0.3 | standalone | 3,984,493 | +0.0497 | 0.6959 | 0.9241 | +0.02 | 0.09 | -1.579 | -1.091 | +1.202 | +1.696 | 0.793 | -1.099 | -1.056 |
| laplace | 0.05 | member | 12,514,286 | -0.0299 | 0.9476 | 0.9575 | +0.16 | 2.79 | -2.554 | -1.543 | +1.540 | +2.644 | 0.716 | -1.305 | -1.365 |
| laplace | 0.05 | standalone | 6,383,457 | +0.0586 | 0.9274 | 0.9413 | +0.39 | 2.91 | -2.297 | -1.366 | +1.641 | +2.773 | 0.718 | -1.284 | -1.344 |
| laplace | 0.1 | member | 13,192,374 | -0.0280 | 0.9129 | 0.9242 | +0.20 | 2.79 | -2.443 | -1.483 | +1.487 | +2.560 | 0.717 | -1.268 | -1.328 |
| laplace | 0.1 | standalone | 5,864,538 | +0.0630 | 0.9248 | 0.9399 | +0.40 | 2.91 | -2.277 | -1.355 | +1.644 | +2.776 | 0.718 | -1.281 | -1.341 |
| laplace | 0.3 | member | 15,152,660 | -0.0200 | 0.7683 | 0.7964 | +0.33 | 2.84 | -1.987 | -1.228 | +1.260 | +2.190 | 0.722 | -1.103 | -1.155 |
| laplace | 0.3 | standalone | 4,605,615 | +0.0660 | 0.9229 | 0.9397 | +0.41 | 2.88 | -2.265 | -1.346 | +1.644 | +2.775 | 0.718 | -1.280 | -1.339 |
| t3 | 0.05 | member | 11,666,781 | -0.0311 | 1.0000 | 0.9588 | +1.35 | 31.87 | -2.550 | -1.436 | +1.423 | +2.794 | 0.672 | -1.294 | -1.419 |
| t3 | 0.05 | standalone | 6,796,092 | +0.0533 | 0.9771 | 0.9455 | +2.15 | 38.74 | -2.210 | -1.283 | +1.518 | +2.943 | 0.674 | -1.273 | -1.396 |
| t3 | 0.1 | member | 12,305,711 | -0.0296 | 0.9709 | 0.9274 | +1.56 | 36.76 | -2.457 | -1.386 | +1.379 | +2.714 | 0.669 | -1.261 | -1.389 |
| t3 | 0.1 | standalone | 6,305,999 | +0.0578 | 0.9719 | 0.9432 | +2.16 | 37.64 | -2.186 | -1.271 | +1.521 | +2.947 | 0.675 | -1.269 | -1.390 |
| t3 | 0.3 | member | 14,094,327 | -0.0230 | 0.8383 | 0.8018 | +2.04 | 41.44 | -2.051 | -1.179 | +1.192 | +2.380 | 0.668 | -1.112 | -1.243 |
| t3 | 0.3 | standalone | 5,154,442 | +0.0629 | 0.9724 | 0.9428 | +2.37 | 42.98 | -2.165 | -1.264 | +1.525 | +2.949 | 0.674 | -1.268 | -1.391 |

## Members by K, prelift (ratios: member / standalone of the same cell)

| Family | α | K | mean K | n | sd | sd ratio | equal weights √(1 − α(2 − α)(1 − 1/K̄)) | actual weights √mean[(1 − α + αw)² + α²(Σw² − w²)] | p1 ratio | p99 ratio | p1 ratio / sd ratio | ex. kurt | MAD/SD | LL Laplace − LL normal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laplace | 0.05 | all | 3.76 | 12,890,373 | 0.973 | 0.963 | 0.964 | 0.966 | 0.945 | 0.981 | 0.982 | 2.83 | 0.714 | +0.063 |
| laplace | 0.05 | 2 | 2.00 | 3,026,114 | 0.981 | 0.971 | 0.975 | 0.975 | 0.951 | 0.992 | 0.980 | 2.86 | 0.713 | +0.064 |
| laplace | 0.05 | 3-4 | 3.48 | 6,137,857 | 0.972 | 0.962 | 0.965 | 0.965 | 0.945 | 0.978 | 0.983 | 2.81 | 0.714 | +0.063 |
| laplace | 0.05 | 5+ | 5.65 | 3,725,187 | 0.968 | 0.958 | 0.959 | 0.959 | 0.939 | 0.975 | 0.981 | 2.85 | 0.714 | +0.063 |
| laplace | 0.1 | all | 4.49 | 13,557,325 | 0.934 | 0.925 | 0.923 | 0.929 | 0.897 | 0.951 | 0.969 | 2.81 | 0.715 | +0.062 |
| laplace | 0.1 | 2 | 2.00 | 2,319,276 | 0.956 | 0.947 | 0.951 | 0.952 | 0.916 | 0.978 | 0.968 | 2.82 | 0.715 | +0.062 |
| laplace | 0.1 | 3-4 | 3.53 | 4,936,961 | 0.937 | 0.928 | 0.929 | 0.931 | 0.902 | 0.951 | 0.972 | 2.80 | 0.715 | +0.062 |
| laplace | 0.1 | 5+ | 6.17 | 6,299,767 | 0.924 | 0.915 | 0.917 | 0.918 | 0.885 | 0.942 | 0.967 | 2.80 | 0.715 | +0.062 |
| laplace | 0.3 | all | 6.98 | 15,441,649 | 0.780 | 0.773 | 0.750 | 0.773 | 0.714 | 0.819 | 0.923 | 2.82 | 0.721 | +0.053 |
| laplace | 0.3 | 2 | 2.00 | 1,339,042 | 0.878 | 0.870 | 0.863 | 0.867 | 0.818 | 0.906 | 0.939 | 2.68 | 0.722 | +0.052 |
| laplace | 0.3 | 3-4 | 3.53 | 2,124,401 | 0.816 | 0.809 | 0.796 | 0.806 | 0.753 | 0.847 | 0.931 | 2.75 | 0.723 | +0.051 |
| laplace | 0.3 | 5+ | 8.15 | 11,977,145 | 0.762 | 0.755 | 0.743 | 0.756 | 0.693 | 0.802 | 0.918 | 2.78 | 0.722 | +0.052 |
| normal | 0.05 | all | 3.58 | 14,156,430 | 0.708 | 0.965 | 0.964 | 0.966 | 0.957 | 0.974 | 0.992 | 0.00 | 0.798 | -0.048 |
| normal | 0.05 | 2 | 2.00 | 3,139,054 | 0.714 | 0.972 | 0.975 | 0.975 | 0.964 | 0.982 | 0.991 | 0.00 | 0.798 | -0.048 |
| normal | 0.05 | 3-4 | 3.48 | 7,832,442 | 0.708 | 0.965 | 0.965 | 0.965 | 0.957 | 0.973 | 0.992 | 0.00 | 0.798 | -0.048 |
| normal | 0.05 | 5+ | 5.41 | 3,184,232 | 0.704 | 0.959 | 0.959 | 0.960 | 0.951 | 0.969 | 0.992 | 0.00 | 0.798 | -0.048 |
| normal | 0.1 | all | 4.33 | 14,913,211 | 0.681 | 0.928 | 0.924 | 0.929 | 0.915 | 0.944 | 0.985 | 0.01 | 0.798 | -0.048 |
| normal | 0.1 | 2 | 2.00 | 2,181,118 | 0.697 | 0.950 | 0.951 | 0.952 | 0.937 | 0.967 | 0.986 | 0.01 | 0.797 | -0.048 |
| normal | 0.1 | 3-4 | 3.55 | 6,241,016 | 0.682 | 0.930 | 0.929 | 0.931 | 0.917 | 0.946 | 0.986 | 0.00 | 0.798 | -0.048 |
| normal | 0.1 | 5+ | 5.85 | 6,490,256 | 0.674 | 0.919 | 0.918 | 0.919 | 0.904 | 0.935 | 0.984 | 0.00 | 0.798 | -0.048 |
| normal | 0.3 | all | 6.67 | 16,748,122 | 0.567 | 0.773 | 0.753 | 0.771 | 0.749 | 0.805 | 0.969 | 0.07 | 0.796 | -0.046 |
| normal | 0.3 | 2 | 2.00 | 1,090,928 | 0.639 | 0.871 | 0.863 | 0.866 | 0.854 | 0.896 | 0.980 | 0.04 | 0.796 | -0.047 |
| normal | 0.3 | 3-4 | 3.59 | 2,185,956 | 0.591 | 0.806 | 0.795 | 0.804 | 0.786 | 0.832 | 0.976 | 0.06 | 0.796 | -0.046 |
| normal | 0.3 | 5+ | 7.55 | 13,470,661 | 0.556 | 0.759 | 0.747 | 0.758 | 0.731 | 0.791 | 0.964 | 0.05 | 0.797 | -0.047 |
| t3 | 0.05 | all | 3.64 | 12,088,305 | 1.081 | 0.930 | 0.964 | 0.967 | 0.927 | 0.995 | 0.997 | 241.53 | 0.643 | +0.167 |
| t3 | 0.05 | 2 | 2.00 | 3,355,228 | 1.079 | 0.929 | 0.975 | 0.975 | 0.927 | 1.004 | 0.998 | 137.87 | 0.648 | +0.159 |
| t3 | 0.05 | 3-4 | 3.46 | 5,568,230 | 1.082 | 0.931 | 0.965 | 0.965 | 0.931 | 0.993 | 1.000 | 250.15 | 0.643 | +0.168 |
| t3 | 0.05 | 5+ | 5.69 | 3,163,699 | 1.081 | 0.931 | 0.959 | 0.959 | 0.919 | 0.987 | 0.988 | 336.02 | 0.639 | +0.174 |
| t3 | 0.1 | all | 4.27 | 12,715,670 | 1.043 | 0.893 | 0.924 | 0.931 | 0.880 | 0.970 | 0.985 | 279.05 | 0.643 | +0.167 |
| t3 | 0.1 | 2 | 2.00 | 2,730,630 | 1.059 | 0.907 | 0.951 | 0.952 | 0.894 | 0.990 | 0.986 | 199.13 | 0.647 | +0.162 |
| t3 | 0.1 | 3-4 | 3.50 | 4,793,311 | 1.048 | 0.898 | 0.930 | 0.931 | 0.882 | 0.969 | 0.982 | 303.74 | 0.640 | +0.172 |
| t3 | 0.1 | 5+ | 6.18 | 5,190,375 | 1.029 | 0.882 | 0.917 | 0.919 | 0.871 | 0.960 | 0.988 | 301.61 | 0.644 | +0.166 |
| t3 | 0.3 | all | 6.50 | 14,432,733 | 0.888 | 0.740 | 0.754 | 0.780 | 0.699 | 0.861 | 0.945 | 295.43 | 0.646 | +0.163 |
| t3 | 0.3 | 2 | 2.00 | 1,746,508 | 0.968 | 0.807 | 0.863 | 0.868 | 0.787 | 0.940 | 0.976 | 98.65 | 0.658 | +0.145 |
| t3 | 0.3 | 3-4 | 3.51 | 2,540,306 | 0.913 | 0.761 | 0.797 | 0.807 | 0.732 | 0.885 | 0.962 | 217.06 | 0.652 | +0.154 |
| t3 | 0.3 | 5+ | 8.03 | 10,144,794 | 0.867 | 0.722 | 0.744 | 0.758 | 0.674 | 0.840 | 0.933 | 369.39 | 0.643 | +0.168 |

## Scale against size: sd by decile of log share at the start of the step

Prelift (all firm-periods, standardized). Last column: the actual-weights reference, D10 over D1.

| Family | α | status | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D10 / D1 | ref_sd D10 / D1 (actual weights) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | member | 0.7011 | 0.7040 | 0.7056 | 0.7060 | 0.7058 | 0.7068 | 0.7091 | 0.7105 | 0.7141 | 0.7198 | 1.027 | 1.022 |
| normal | 0.05 | standalone | 0.7324 | 0.7317 | 0.7338 | 0.7361 | 0.7354 | 0.7345 | 0.7343 | 0.7341 | 0.7345 | 0.7322 | 1.000 | 1.000 |
| normal | 0.1 | member | 0.6717 | 0.6749 | 0.6749 | 0.6763 | 0.6768 | 0.6776 | 0.6798 | 0.6837 | 0.6896 | 0.7021 | 1.045 | 1.043 |
| normal | 0.1 | standalone | 0.7310 | 0.7314 | 0.7325 | 0.7355 | 0.7355 | 0.7339 | 0.7347 | 0.7336 | 0.7336 | 0.7321 | 1.001 | 1.000 |
| normal | 0.3 | member | 0.5573 | 0.5568 | 0.5566 | 0.5566 | 0.5574 | 0.5594 | 0.5624 | 0.5684 | 0.5796 | 0.6111 | 1.096 | 1.109 |
| normal | 0.3 | standalone | 0.7299 | 0.7302 | 0.7319 | 0.7345 | 0.7358 | 0.7350 | 0.7359 | 0.7343 | 0.7339 | 0.7338 | 1.005 | 1.000 |
| laplace | 0.05 | member | 0.9599 | 0.9622 | 0.9668 | 0.9706 | 0.9718 | 0.9730 | 0.9745 | 0.9776 | 0.9797 | 0.9907 | 1.032 | 1.023 |
| laplace | 0.05 | standalone | 1.0054 | 1.0092 | 1.0082 | 1.0124 | 1.0149 | 1.0150 | 1.0136 | 1.0099 | 1.0080 | 1.0069 | 1.002 | 1.000 |
| laplace | 0.1 | member | 0.9194 | 0.9223 | 0.9257 | 0.9266 | 0.9300 | 0.9320 | 0.9346 | 0.9402 | 0.9452 | 0.9655 | 1.050 | 1.044 |
| laplace | 0.1 | standalone | 1.0064 | 1.0055 | 1.0054 | 1.0119 | 1.0150 | 1.0123 | 1.0149 | 1.0086 | 1.0112 | 1.0081 | 1.002 | 1.000 |
| laplace | 0.3 | member | 0.7678 | 0.7662 | 0.7665 | 0.7674 | 0.7673 | 0.7707 | 0.7745 | 0.7830 | 0.7951 | 0.8385 | 1.092 | 1.105 |
| laplace | 0.3 | standalone | 1.0052 | 1.0024 | 1.0027 | 1.0115 | 1.0128 | 1.0118 | 1.0113 | 1.0096 | 1.0103 | 1.0090 | 1.004 | 1.000 |
| t3 | 0.05 | member | 1.0628 | 1.0623 | 1.0686 | 1.0837 | 1.0834 | 1.0831 | 1.0769 | 1.0986 | 1.0960 | 1.0917 | 1.027 | 1.024 |
| t3 | 0.05 | standalone | 1.1568 | 1.1585 | 1.1602 | 1.1745 | 1.1634 | 1.1606 | 1.1450 | 1.1596 | 1.1573 | 1.1820 | 1.022 | 1.000 |
| t3 | 0.1 | member | 1.0220 | 1.0132 | 1.0303 | 1.0369 | 1.0510 | 1.0291 | 1.0426 | 1.0687 | 1.0603 | 1.0705 | 1.048 | 1.046 |
| t3 | 0.1 | standalone | 1.1716 | 1.1673 | 1.1712 | 1.1838 | 1.1667 | 1.1574 | 1.1695 | 1.1372 | 1.1801 | 1.1666 | 0.996 | 1.000 |
| t3 | 0.3 | member | 0.8787 | 0.8686 | 0.8655 | 0.8808 | 0.8767 | 0.8764 | 0.8966 | 0.8914 | 0.9047 | 0.9378 | 1.067 | 1.108 |
| t3 | 0.3 | standalone | 1.1680 | 1.2052 | 1.2277 | 1.1942 | 1.2099 | 1.2071 | 1.1849 | 1.1832 | 1.2083 | 1.2134 | 1.039 | 1.000 |

The brief's within-market version (standardized, lift periods excluded): the standalones' slope is the
exclusion's truncation of small firms near the floor, not volatility.

| Family | α | status | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D10 / D1 | ref_sd D10 / D1 (actual weights) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | member | 0.6270 | 0.6943 | 0.7042 | 0.7052 | 0.7055 | 0.7060 | 0.7085 | 0.7100 | 0.7128 | 0.7147 | 1.140 | 1.022 |
| normal | 0.05 | standalone | 0.5335 | 0.5998 | 0.6485 | 0.6863 | 0.7139 | 0.7264 | 0.7322 | 0.7340 | 0.7337 | 0.7273 | 1.363 | 1.000 |
| normal | 0.1 | member | 0.6079 | 0.6675 | 0.6739 | 0.6759 | 0.6760 | 0.6771 | 0.6790 | 0.6831 | 0.6884 | 0.6967 | 1.146 | 1.043 |
| normal | 0.1 | standalone | 0.5283 | 0.5933 | 0.6425 | 0.6815 | 0.7082 | 0.7245 | 0.7310 | 0.7336 | 0.7327 | 0.7289 | 1.380 | 1.000 |
| normal | 0.3 | member | 0.5228 | 0.5549 | 0.5558 | 0.5558 | 0.5567 | 0.5588 | 0.5617 | 0.5677 | 0.5785 | 0.6063 | 1.160 | 1.111 |
| normal | 0.3 | standalone | 0.5183 | 0.5871 | 0.6383 | 0.6782 | 0.7079 | 0.7249 | 0.7324 | 0.7341 | 0.7332 | 0.7336 | 1.415 | 1.000 |
| laplace | 0.05 | member | 0.8012 | 0.8908 | 0.9378 | 0.9578 | 0.9674 | 0.9695 | 0.9730 | 0.9752 | 0.9769 | 0.9803 | 1.224 | 1.023 |
| laplace | 0.05 | standalone | 0.7591 | 0.7868 | 0.8286 | 0.8730 | 0.9142 | 0.9486 | 0.9767 | 0.9941 | 1.0057 | 0.9995 | 1.317 | 1.000 |
| laplace | 0.1 | member | 0.7785 | 0.8640 | 0.9037 | 0.9187 | 0.9265 | 0.9295 | 0.9329 | 0.9382 | 0.9428 | 0.9547 | 1.226 | 1.045 |
| laplace | 0.1 | standalone | 0.7557 | 0.7829 | 0.8233 | 0.8657 | 0.9076 | 0.9427 | 0.9721 | 0.9948 | 1.0049 | 1.0027 | 1.327 | 1.000 |
| laplace | 0.3 | member | 0.6817 | 0.7443 | 0.7606 | 0.7647 | 0.7662 | 0.7682 | 0.7732 | 0.7811 | 0.7928 | 0.8292 | 1.216 | 1.108 |
| laplace | 0.3 | standalone | 0.7455 | 0.7761 | 0.8177 | 0.8622 | 0.9037 | 0.9379 | 0.9696 | 0.9922 | 1.0039 | 1.0077 | 1.352 | 1.000 |
| t3 | 0.05 | member | 0.8640 | 0.9214 | 0.9617 | 0.9886 | 1.0037 | 1.0191 | 1.0281 | 1.0473 | 1.0506 | 1.0646 | 1.232 | 1.024 |
| t3 | 0.05 | standalone | 0.8673 | 0.8603 | 0.8879 | 0.9143 | 0.9462 | 0.9786 | 1.0041 | 1.0253 | 1.0484 | 1.0779 | 1.243 | 1.000 |
| t3 | 0.1 | member | 0.8485 | 0.8994 | 0.9364 | 0.9591 | 0.9746 | 0.9827 | 0.9932 | 1.0145 | 1.0211 | 1.0359 | 1.221 | 1.047 |
| t3 | 0.1 | standalone | 0.8608 | 0.8530 | 0.8818 | 0.9090 | 0.9381 | 0.9709 | 1.0006 | 1.0187 | 1.0521 | 1.0672 | 1.240 | 1.000 |
| t3 | 0.3 | member | 0.7585 | 0.7950 | 0.8182 | 0.8291 | 0.8381 | 0.8445 | 0.8515 | 0.8584 | 0.8671 | 0.8992 | 1.185 | 1.112 |
| t3 | 0.3 | standalone | 0.8526 | 0.8512 | 0.8773 | 0.9175 | 0.9330 | 0.9704 | 1.0023 | 1.0203 | 1.0437 | 1.0717 | 1.257 | 1.000 |

Raw Δ log s (lift periods excluded): the same truncation at the bottom, and the market term lowers the top decile.

| Family | α | status | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D10 / D1 | ref_sd D10 / D1 (actual weights) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| normal | 0.05 | member | 0.1358 | 0.1547 | 0.1582 | 0.1584 | 0.1581 | 0.1579 | 0.1579 | 0.1568 | 0.1550 | 0.1412 | 1.040 | 1.022 |
| normal | 0.05 | standalone | 0.1122 | 0.1262 | 0.1395 | 0.1509 | 0.1597 | 0.1646 | 0.1670 | 0.1674 | 0.1662 | 0.1421 | 1.267 | 1.000 |
| normal | 0.1 | member | 0.1324 | 0.1494 | 0.1516 | 0.1517 | 0.1515 | 0.1511 | 0.1509 | 0.1505 | 0.1491 | 0.1347 | 1.017 | 1.043 |
| normal | 0.1 | standalone | 0.1114 | 0.1247 | 0.1374 | 0.1485 | 0.1574 | 0.1632 | 0.1658 | 0.1666 | 0.1659 | 0.1544 | 1.386 | 1.000 |
| normal | 0.3 | member | 0.1154 | 0.1243 | 0.1242 | 0.1237 | 0.1234 | 0.1233 | 0.1233 | 0.1235 | 0.1237 | 0.1158 | 1.004 | 1.111 |
| normal | 0.3 | standalone | 0.1093 | 0.1223 | 0.1344 | 0.1453 | 0.1539 | 0.1594 | 0.1622 | 0.1629 | 0.1619 | 0.1594 | 1.458 | 1.000 |
| laplace | 0.05 | member | 0.1700 | 0.1909 | 0.2034 | 0.2090 | 0.2112 | 0.2116 | 0.2116 | 0.2105 | 0.2070 | 0.1850 | 1.088 | 1.023 |
| laplace | 0.05 | standalone | 0.1572 | 0.1649 | 0.1757 | 0.1879 | 0.1990 | 0.2094 | 0.2179 | 0.2234 | 0.2265 | 0.2067 | 1.314 | 1.000 |
| laplace | 0.1 | member | 0.1658 | 0.1858 | 0.1968 | 0.2007 | 0.2023 | 0.2025 | 0.2025 | 0.2021 | 0.1991 | 0.1773 | 1.069 | 1.045 |
| laplace | 0.1 | standalone | 0.1565 | 0.1639 | 0.1742 | 0.1853 | 0.1966 | 0.2070 | 0.2156 | 0.2222 | 0.2261 | 0.2174 | 1.390 | 1.000 |
| laplace | 0.3 | member | 0.1460 | 0.1610 | 0.1653 | 0.1660 | 0.1659 | 0.1658 | 0.1659 | 0.1663 | 0.1660 | 0.1544 | 1.058 | 1.108 |
| laplace | 0.3 | standalone | 0.1545 | 0.1612 | 0.1710 | 0.1817 | 0.1924 | 0.2018 | 0.2103 | 0.2168 | 0.2201 | 0.2190 | 1.418 | 1.000 |
| t3 | 0.05 | member | 0.1820 | 0.1965 | 0.2073 | 0.2139 | 0.2181 | 0.2214 | 0.2239 | 0.2260 | 0.2254 | 0.2056 | 1.130 | 1.024 |
| t3 | 0.05 | standalone | 0.1767 | 0.1800 | 0.1885 | 0.1967 | 0.2073 | 0.2156 | 0.2226 | 0.2293 | 0.2343 | 0.2252 | 1.275 | 1.000 |
| t3 | 0.1 | member | 0.1784 | 0.1922 | 0.2018 | 0.2075 | 0.2114 | 0.2137 | 0.2158 | 0.2186 | 0.2195 | 0.1990 | 1.115 | 1.047 |
| t3 | 0.1 | standalone | 0.1762 | 0.1786 | 0.1865 | 0.1957 | 0.2047 | 0.2132 | 0.2213 | 0.2269 | 0.2342 | 0.2313 | 1.313 | 1.000 |
| t3 | 0.3 | member | 0.1596 | 0.1700 | 0.1764 | 0.1788 | 0.1809 | 0.1826 | 0.1840 | 0.1842 | 0.1835 | 0.1700 | 1.065 | 1.112 |
| t3 | 0.3 | standalone | 0.1741 | 0.1768 | 0.1842 | 0.1939 | 0.2005 | 0.2106 | 0.2177 | 0.2232 | 0.2296 | 0.2354 | 1.352 | 1.000 |

## Figure

`figures/appendix/growth_density_alpha0.3.pdf`: α = 0.3, one panel per family, density of the prelift measure for
standalones and members on a log density axis, each with its ML Laplace fit (location the median, scale the mean
absolute deviation from it).

## Tests

`tests/test_growth_distributions.py`:
- a synthetic trace with i.i.d. Laplace shocks and α = 0 reproduces the shocks' sd, IQR, MAD/SD, kurtosis and
  skewness within sampling error under all three excluded-sample measures;
- on a hand-built two-firm unit, the pooling kernel's update is log(1 + δᵢ) with δᵢ = (1 − α) rᵢ + α Σ wⱼ rⱼ − Φ(2),
  and the script returns it as the member's growth;
- member status, K and the exclusions on a hand-built trace; demeaning cancels a market-common term; prelift
  adds the lifted periods back and the reference variance is 1 − α(2 − α)(1 − 1/K) with equal weights;
- the output CSV's K bins and deciles partition the status rows.
