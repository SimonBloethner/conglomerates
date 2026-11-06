**Where scope starts to fall, and whether cost binds at K\* (C35).** There is no threshold: scope falls at every
cost level tried, by about 0.35 in mean K per doubling of c0.

- For laplace at α = 0.3, mean K (medians over reps) is 4.33, 4.00 and 3.67 at ×0.5, ×1 and ×2 (C33 `cost-level`
  block and the main cell, which use different seeds), then 3.14, 2.76 and 2.45 at ×5, ×10 and ×20.
- For t3 at α = 0.3 it is 3.67, 3.04, 2.77 and 2.50 at ×1, 5, 10 and 20.

The fall comes through acceptance at every K ≥ 3. For laplace at α = 0.3 and K = 4, acceptance is 0.41, 0.28,
0.20 and 0.11 at ×1, 5, 10 and 20 (`pilot_c/acceptance_by_K_pooled.csv`). Mergers per period barely move, because
more standalones make more proposals.

K\* falls faster than K: K/K\* rises from 0.25 (laplace) and 0.08 (t3) at ×1 to 0.61 and 0.50 at ×20. At ×20, K\*
(4 for laplace, 5 for t3) has come down to the scope that forms at ×1, and the analytical marginal cost of the
K-th member is close to its marginal gain at K ≈ 4–5. Yet K is 2.45–2.50. This is still half to two thirds of
K\*, against 0.7 at α = 0.1, where K\* = 3.

Cost therefore binds at the margin from the lowest level on, but even at ×20 it does not take over: K does not
settle at K\*, and attrition and consent keep it below. (K_mean counts units only, so it cannot fall below 2.)
