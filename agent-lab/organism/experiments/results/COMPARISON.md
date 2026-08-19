# Experiment Comparison Report

## Random Policy

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Final Survivors | 0.200 | 0.400 | 0.000 | 1.000 |
| Avg Energy (final) | 4.594 | 9.207 | 0.000 | 24.321 |
| Largest Cluster | 0.200 | 0.400 | 0.000 | 1.000 |
| Signal Diversity | 0.000 | 0.000 | 0.000 | 0.000 |
| Spatial Entropy | 0.000 | 0.000 | 0.000 | 0.000 |
| Resource Consumed | 7154.414 | 770.851 | 6124.770 | 8842.631 |
| Survivors / 1000 Resource | 0.026 | 0.053 | 0.000 | 0.139 |
| Elapsed (s) | 4.056 | 0.136 | 3.820 | 4.220 |

## Rule_based Policy

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Final Survivors | 46.300 | 2.492 | 43.000 | 50.000 |
| Avg Energy (final) | 20.890 | 0.444 | 20.233 | 21.896 |
| Largest Cluster | 2.800 | 1.166 | 2.000 | 6.000 |
| Signal Diversity | 0.993 | 0.006 | 0.986 | 1.003 |
| Spatial Entropy | 4.961 | 0.106 | 4.772 | 5.124 |
| Resource Consumed | 38168.636 | 1372.984 | 35697.377 | 40251.824 |
| Survivors / 1000 Resource | 1.212 | 0.027 | 1.168 | 1.271 |
| Elapsed (s) | 10.991 | 0.411 | 10.340 | 11.680 |

## Cross-Policy Comparison

| Metric | Random | Rule_based |
|--------|---|--|
| Final Survivors | 0.20 ± 0.40 | 46.30 ± 2.49 |
| Avg Time (s) | 4.06 ± 0.14 | 10.99 ± 0.41 |

## Interpretation Boundaries

**DESIGNED:** metabolism, resource regeneration, action costs, collision handling, and each policy's action rules.

**OBSERVED:** the tables contain final-step measurements over ten matched seeds.

**INFERRED:** the consume-and-search rule improves survival under this resource model. This does not establish intelligence or self-organization.

**SPECULATIVE:** signal semantics, role differentiation, and organism-like behavior are not demonstrated by these metrics.