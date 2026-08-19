# Brain Architecture: Deterministic Pilot

## Status

This is a harness and causal-path pilot using a deterministic test double. It is **not** evidence that an LLM brain, personality, subjectivity, or emergent intelligence has been created.

## Design

3 matched seeds × 20 instances × 4 task types. All conditions received the exact same immutable TaskView objects.

| Condition | Accuracy | Switch | Recall | Fault recovery | Calls | Input units |
|---|---:|---:|---:|---:|---:|---:|
| single_long_context | 0.500 | 0.000 | 1.000 | 0.000 | 1.00 | 6.25 |
| sequential_reflection | 0.750 | 1.000 | 1.000 | 0.000 | 4.00 | 17.50 |
| flat_ensemble | 0.967 | 1.000 | 0.867 | 1.000 | 4.00 | 9.25 |
| brain_full | 1.000 | 1.000 | 1.000 | 1.000 | 3.75 | 13.25 |
| brain_no_memory | 0.383 | 1.000 | 0.283 | 0.250 | 3.75 | 10.00 |
| brain_no_error_monitor | 0.750 | 1.000 | 1.000 | 0.000 | 3.00 | 8.75 |
| brain_static_routing | 1.000 | 1.000 | 1.000 | 1.000 | 4.00 | 14.50 |
| brain_no_internal_state_coupling | 0.750 | 1.000 | 1.000 | 0.000 | 3.00 | 8.75 |

## Paired Seed Contrasts

Differences are left minus right. With only three pilot seeds, exact p-values are coarse and are included to validate the analysis path, not to support formal claims.

| Contrast | Accuracy Δ | 95% bootstrap CI | exact p | Calls Δ |
|---|---:|---:|---:|---:|
| brain_full − single_long_context | +0.500 | [+0.500, +0.500] | 0.250 | +2.75 |
| brain_full − sequential_reflection | +0.250 | [+0.250, +0.250] | 0.250 | -0.25 |
| brain_full − flat_ensemble | +0.033 | [+0.025, +0.050] | 0.250 | -0.25 |
| brain_full − brain_no_memory | +0.617 | [+0.600, +0.637] | 0.250 | +0.00 |
| brain_full − brain_no_error_monitor | +0.250 | [+0.250, +0.250] | 0.250 | +0.75 |
| brain_full − brain_static_routing | +0.000 | [+0.000, +0.000] | 1.000 | -0.25 |
| brain_full − brain_no_internal_state_coupling | +0.250 | [+0.250, +0.250] | 0.250 | +0.75 |

## History-dependence manipulation

- Different-history divergence: 1
- Divergence after memory reset: 0
- Choice followed swapped memory: 1
- Interpretation: this only verifies that memory is a causal path for history-dependent choices. The histories and probe were deliberately constructed, so this is not emergent individuality.

## Interpretation discipline

### DESIGNED

- Bounded workspace, six-record retrieval, conflict-triggered robust processing, and robust-output gating are explicit architecture choices.
- Task families were constructed to exercise those paths.

### OBSERVED

- The table above records deterministic harness behavior and compute use.
- Reset and swap interventions change the history-probe result through memory.

### INFERRED

- A selective ablation difference indicates that the implemented path is active and measurable in this harness.

### SPECULATIVE / NOT YET TESTED

- Whether the architecture helps a real language model under a strict token-matched budget.
- Whether stable behavioral differentiation appears without an explicitly constructed history cue.
- Any claim of personality, subjectivity, consciousness, biological similarity, or emergent intelligence.

## Next gate

Freeze one real ModelAdapter, a token-matched matrix, and a blind procedural evaluation set. Do not reuse this pilot's tuned task instances as formal evidence.
