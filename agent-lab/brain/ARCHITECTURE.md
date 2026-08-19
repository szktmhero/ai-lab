# Brain Kernel Architecture

## Boundary of the experiment

The system under test receives a `TaskView`. It never receives a `Task`, which is
the evaluator-owned object containing `correct_answer` and experimental
annotations. The event observer is append-only and is not read by the system.

```text
Task(correct answer, metadata)
  |
  +-- evaluator only
  |
  +-- TaskView(observations, query) --> system --> SystemResult
                                             |
                                             +--> evaluator metrics
```

## Episode lifecycle

1. `PerceptionGateway` converts each public `EvidenceEvent` into a `Signal`.
2. The signal enters a capacity-four, TTL-three `BoundedWorkspace`.
3. If enabled, salient evidence is copied to `EpisodicMemory` with provenance.
4. `ErrorMonitor` measures disagreement among public evidence winners.
5. The dynamic `BrainState` records novelty, load, uncertainty, prediction error,
   and resource pressure. Novelty can force a memory write; load and uncertainty
   control retrieval; prediction error or sustained novelty can request robust
   processing; high resource pressure suppresses that optional call.
6. `SalienceRouter` selects the normal or robust inference path. It does not see
   evidence scores or evaluator state itself.
7. Fast and deliberate outputs become bounded hypothesis signals.
8. When conflict crosses the declared threshold, robust output is additionally
   produced and promoted at the action interface.
9. The final action includes all source IDs used by the upstream model calls.

## Causal paths and ablations

| Path | Full behavior | Ablation |
|---|---|---|
| evidence → episodic memory → cortex | Retrieve six newest cue matches | `brain_no_memory` |
| evidence disagreement → prediction error | Update state | `brain_no_error_monitor` |
| prediction error → robust routing | Activate robust cortex | `brain_no_internal_state_coupling` |
| adaptive state → module activation | Skip unnecessary robust calls | `brain_static_routing` |

An ablation must remove the actual edge, not merely hide a metric or rename a
module. Tests assert both the trace and the downstream decision.

## Model invariance

Every condition uses `DeterministicModelAdapter` in the pilot. The adapter has no
memory, random state, condition name, task answer, or evaluator reference. Its
four modes are small evidence aggregation operations. Architecture controls
which public signals reach which mode; it does not replace the adapter.

Phase 3 implements `OpenAIResponsesAdapter` behind the same
`ModelAdapter.infer` contract. It aliases signal IDs before prompting, hashes
provenance roots, ignores dynamic state, exposes no condition or evaluator data,
uses no conversation state, and returns schema-constrained decisions. All eight
conditions share one adapter instance and the same immutable token limits.

## Persistent memory and individuality tests

The `BrainSystem` memory persists across episodes within one seed and condition.
Standard benchmark cues are unique to prevent accidental cross-task transfer.
The separate history-dependence manipulation deliberately reuses one cue.

That manipulation applies three interventions:

- train two otherwise identical systems on different evidence histories;
- repeat a shared ambiguous probe after cloning each history;
- reset and swap the memory records.

A choice that follows reset and swap validates the memory path. Because both the
history and cue are designed, it is not classified as emergent personality.

## Budget semantics

`InferenceBudget.ensure_can_consume` checks limits without mutation and
`InferenceBudget.consume` commits successful usage. The deterministic pilot unit
is one structured signal.

For Phase 3, the adapter sends the same prompt and schema payload to the
provider's `responses.input_tokens.count` endpoint before inference. It rejects a
call unless the counted input plus the entire per-call output allowance fits.
After completion, the preflight input count must exactly equal
`response.usage.input_tokens`; actual input and output tokens are then committed.
Any mismatch aborts the run. Conditions share identical hard caps but are not
padded with dummy work, because adaptive compute reduction is itself H4.

The protocol lock freezes four calls, 8,192 input tokens, 512 output tokens, and
128 output tokens per call. Cached tokens remain part of provider-reported input
usage rather than being silently removed from the compute comparison.

## Trace contract

Every result contains:

- final candidate and confidence;
- upstream provenance;
- calls and input/output units;
- usage unit (`signals` or `tokens`), provider response/model IDs, and request and
  structured-output hashes;
- activated modules and router reason;
- final dynamic state;
- fast, deliberate, and optional robust candidates;
- workspace turnover and utilization;
- memory writes, retrievals, hits, and retrieved record count;
- invalid-action and no-op counts.

The JSONL event log adds perception, routing, model-output, and final-action
events. Its sequential `event_index` allows an episode to be reconstructed without
feeding the log back into the decision loop.
