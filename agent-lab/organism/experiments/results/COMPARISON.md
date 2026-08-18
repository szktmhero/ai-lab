# Experiment Comparison Report

## Random Policy

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Survival Rate | 0.300 | 0.458 | 0.000 | 1.000 |
| Avg Energy (final) | 21.918 | 35.076 | 0.000 | 99.200 |
| Largest Cluster | 0.300 | 0.458 | 0.000 | 1.000 |
| Signal Diversity | 0.000 | 0.000 | 0.000 | 0.000 |
| Spatial Entropy | 0.000 | 0.000 | 0.000 | 0.000 |
| Elapsed (s) | 4.408 | 0.252 | 4.040 | 4.860 |

## Rule_based Policy

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Survival Rate | 39.400 | 3.412 | 34.000 | 46.000 |
| Avg Energy (final) | 21.218 | 0.510 | 20.331 | 21.724 |
| Largest Cluster | 2.300 | 0.458 | 2.000 | 3.000 |
| Signal Diversity | 1.026 | 0.002 | 1.022 | 1.030 |
| Spatial Entropy | 4.763 | 0.104 | 4.615 | 4.985 |
| Elapsed (s) | 13.078 | 0.839 | 11.810 | 14.400 |

## Cross-Policy Comparison

| Metric | Random | Rule_based |
|--------|---|--|
| Survival Rate | 0.30 ± 0.46 | 39.40 ± 3.41 |
| Avg Time (s) | 4.41 ± 0.25 | 13.08 ± 0.84 |