# Cached baseline rare-event sweep

- Completed: 10/10 cases
- Scores are unweighted case means; this is not a full DAVIS benchmark.

| Method | Cases | J&F | Visible J&F | Past backbone | Wall s |
|---|---:|---:|---:|---:|---:|
| Direct Copy | 10 | 0.278243 | 0.000000 | 0.00 | 24.893 |
| Target Reset (blank object slot) | 10 | 0.274672 | 0.000000 | 1.00 | 24.079 |
| Last-Mask | 10 | 0.468188 | 0.319945 | 1.00 | 21.482 |
| Replay-1 | 10 | 0.468188 | 0.319945 | 1.00 | 21.729 |
| Replay-2 | 10 | 0.538350 | 0.425541 | 2.00 | 22.744 |
| Replay-4 | 10 | 0.700364 | 0.623753 | 4.00 | 23.432 |
| Full Replay / Large-native | 10 | 0.603838 | 0.483535 | 30.50 | 33.545 |

## Post-switch temporal metrics

| Method | +1 J&F | +5 J&F | +20 J&F | Shock first 5 visible | Identity-loss rate | Recovery rate | Mean recovery frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct Copy | 0.000000 | 0.000000 | 0.000000 | 0.635878 | 0.700 | 0.000 | n/a |
| Target Reset (blank object slot) | 0.000000 | 0.000000 | 0.000000 | 0.635878 | 0.700 | 0.000 | n/a |
| Last-Mask | 0.352143 | 0.451538 | 0.333493 | 0.283680 | 0.500 | 0.300 | 2.000000 |
| Replay-1 | 0.352143 | 0.451538 | 0.333493 | 0.283680 | 0.500 | 0.300 | 2.000000 |
| Replay-2 | 0.443937 | 0.670875 | 0.412809 | 0.165450 | 0.300 | 0.500 | 2.000000 |
| Replay-4 | 0.534164 | 0.790990 | 0.653715 | 0.025282 | 0.100 | 0.600 | 2.666667 |
| Full Replay / Large-native | 0.597803 | 0.710352 | 0.417813 | 0.000000 | 0.000 | 0.700 | 1.857143 |
