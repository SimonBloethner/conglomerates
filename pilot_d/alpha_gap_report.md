# C43: main block at α ∈ {0.6, 0.7, 0.8}

Pilot D, moment rule, power law, main IQR range, T = 11,000, retained window 8,000–11,000. Medians over 5 reps (the same seeds in every row of a family). α = 0.5: block `main`; α = 0.6–0.8: block `alpha-gap` (ids 2100–2144); α = 0.9: block `alpha-0.9` (ids 1995–2009). Per-rep values: `pilot_d/alpha_gap.csv`.

## normal

| α | cell | K | pooled share | Hill | HHI within | floor exits / period | voluntary exits / period | acceptance | floor hit (member) | floor hit (standalone) | top 10% share | K_eff / K | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | main | 7.70 | 0.956 | 1.342 | 0.109 | 15.59 | 0.000 | 0.269 | 0.0079 | 0.1778 | 0.647 | 0.648 | 239 |
| 0.6 | alpha-gap | 9.30 | 0.963 | 1.438 | 0.096 | 11.12 | 0.000 | 0.253 | 0.0054 | 0.1724 | 0.619 | 0.639 | 286 |
| 0.7 | alpha-gap | 10.92 | 0.969 | 1.553 | 0.084 | 7.59 | 0.000 | 0.235 | 0.0035 | 0.1645 | 0.602 | 0.644 | 327 |
| 0.8 | alpha-gap | 13.12 | 0.976 | 1.691 | 0.070 | 4.57 | 0.000 | 0.195 | 0.0020 | 0.1519 | 0.576 | 0.663 | 384 |
| 0.9 | alpha-0.9 | 15.99 | 0.980 | 1.978 | 0.056 | 2.44 | 0.000 | 0.155 | 0.0011 | 0.1363 | 0.531 | 0.711 | 445 |

## laplace

| α | cell | K | pooled share | Hill | HHI within | floor exits / period | voluntary exits / period | acceptance | floor hit (member) | floor hit (standalone) | top 10% share | K_eff / K | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | main | 7.76 | 0.934 | 1.344 | 0.113 | 24.49 | 0.000 | 0.379 | 0.0135 | 0.2060 | 0.738 | 0.643 | 230 |
| 0.6 | alpha-gap | 9.34 | 0.945 | 1.417 | 0.101 | 18.39 | 0.000 | 0.358 | 0.0096 | 0.2029 | 0.733 | 0.637 | 285 |
| 0.7 | alpha-gap | 11.54 | 0.953 | 1.540 | 0.086 | 12.91 | 0.000 | 0.326 | 0.0064 | 0.1965 | 0.701 | 0.632 | 342 |
| 0.8 | alpha-gap | 14.07 | 0.959 | 1.707 | 0.070 | 8.46 | 0.000 | 0.273 | 0.0040 | 0.1863 | 0.687 | 0.646 | 420 |
| 0.9 | alpha-0.9 | 17.20 | 0.966 | 1.942 | 0.058 | 4.90 | 0.000 | 0.215 | 0.0022 | 0.1748 | 0.671 | 0.685 | 521 |

## t3

| α | cell | K | pooled share | Hill | HHI within | floor exits / period | voluntary exits / period | acceptance | floor hit (member) | floor hit (standalone) | top 10% share | K_eff / K | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | main | 6.78 | 0.920 | 1.221 | 0.136 | 31.15 | 0.000 | 0.409 | 0.0186 | 0.2187 | 0.790 | 0.675 | 219 |
| 0.6 | alpha-gap | 8.03 | 0.930 | 1.308 | 0.121 | 25.03 | 0.000 | 0.389 | 0.0143 | 0.2162 | 0.796 | 0.669 | 262 |
| 0.7 | alpha-gap | 9.60 | 0.936 | 1.428 | 0.103 | 19.67 | 0.000 | 0.365 | 0.0107 | 0.2157 | 0.791 | 0.666 | 310 |
| 0.8 | alpha-gap | 11.57 | 0.945 | 1.563 | 0.086 | 14.47 | 0.000 | 0.331 | 0.0075 | 0.2093 | 0.804 | 0.674 | 369 |
| 0.9 | alpha-0.9 | 13.94 | 0.950 | 1.783 | 0.069 | 10.55 | 0.000 | 0.298 | 0.0053 | 0.2016 | 0.811 | 0.683 | 450 |
