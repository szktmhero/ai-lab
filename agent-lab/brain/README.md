# Brain Architecture Experiment

> How much intelligence comes from the model, and how much comes from the
> architecture around it?

This directory contains the first causal Brain Kernel: a bounded workspace,
episodic memory, conflict monitoring, dynamic internal state, salience routing,
and a replaceable model adapter. The initial run uses a deterministic test double
to validate experimental controls before any real-model claim is attempted.

## Quick start

From the repository root:

```bash
PYTHONPATH=agent-lab python -m brain.experiments.run
```

Tests, including the existing organism and society suites:

```bash
UV_CACHE_DIR=/tmp/ai-lab-uv-cache PYTHONPATH=agent-lab \
  uv run --with pytest --with numpy python -m pytest \
  agent-lab/organism/tests agent-lab/society/tests agent-lab/brain/tests -q
```

## Conditions

- `single_long_context`
- `sequential_reflection`
- `flat_ensemble`
- `brain_full`
- `brain_no_memory`
- `brain_no_error_monitor`
- `brain_static_routing`
- `brain_no_internal_state_coupling`

All conditions share the same adapter and immutable public task views. The
evaluator owns answers and fault annotations.

## Results and interpretation

The canonical pilot is three matched seeds × 20 instances × four task types.

| Condition | Accuracy | Switch | Recall | Fault recovery | Calls |
|---|---:|---:|---:|---:|---:|
| `single_long_context` | 0.500 | 0.000 | 1.000 | 0.000 | 1.00 |
| `sequential_reflection` | 0.750 | 1.000 | 1.000 | 0.000 | 4.00 |
| `flat_ensemble` | 0.967 | 1.000 | 0.867 | 1.000 | 4.00 |
| `brain_full` | 1.000 | 1.000 | 1.000 | 1.000 | 3.75 |
| `brain_no_memory` | 0.383 | 1.000 | 0.283 | 0.250 | 3.75 |
| `brain_no_error_monitor` | 0.750 | 1.000 | 1.000 | 0.000 | 3.00 |
| `brain_static_routing` | 1.000 | 1.000 | 1.000 | 1.000 | 4.00 |
| `brain_no_internal_state_coupling` | 0.750 | 1.000 | 1.000 | 0.000 | 3.00 |

**Observed:** the targeted memory and error-routing paths are active, and adaptive
routing saved 0.25 calls per episode relative to the fixed route at equal pilot
accuracy. The flat ensemble was already close to the full architecture, so the
pilot does not establish a distinctive Brain Architecture advantage.

The history manipulation produced different common-probe choices, removed the
difference on memory reset, and reversed it on memory swap. This validates memory
as a causal path for history dependence; the manipulation was explicitly designed
and is not emergent individuality.

See `experiments/results/COMPARISON.md` for the full generated report and paired
pilot contrasts.

The deterministic pilot can establish that a causal route is implemented and
measurable. It cannot establish that the architecture is brain-like, that it
improves a real LLM, or that individuality, subjectivity, or emergence occurred.

See [the research specification](../../lab_brain.md) and
[the technical architecture](ARCHITECTURE.md) before interpreting results.
