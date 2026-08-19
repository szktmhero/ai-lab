# Phase 3: Locked Real-Model Protocol

## Status

**IMPLEMENTED / LOCKED / NOT EXECUTED** as of 2026-08-20.

Phase 3 converts the deterministic causal-path harness into a real-model-ready
experiment without yet spending money or revealing blind outcomes. The offline
preflight validates the committed protocol, prompt fingerprint, task-suite hash,
runtime source hash, matrix size, and execution gate. It does not call OpenAI.

## Frozen design

| Item | Locked value |
|---|---|
| Protocol | `brain-real-llm-v1` |
| Adapter | `OpenAIResponsesAdapter` |
| Model | `gpt-5.4-mini-2026-03-17` |
| OpenAI Python SDK | `2.51.0` |
| Reasoning | `none` |
| Tools / state | none / no prior-response chain |
| Client transport | retries `0`, timeout `120s`, minimum `0.15s` between request starts |
| Per-call output cap | 128 tokens |
| Episode cap | 4 calls, 8,192 input tokens, 512 output tokens |
| Blind suite | 10 seeds × 20 instances × 4 task types = 800 tasks |
| Matrix | 800 tasks × 8 conditions = 6,400 episodes |
| Inference unit | paired per-seed mean |

The worst-case matrix permits 25,600 inference calls and an equal number of
token-count requests. Fixed pacing therefore implies a request-start span of at
least about 128 minutes before model latency; this is a formal run, not a quick
smoke test.

Every condition receives the same cap and one condition-blind adapter. Actual
usage is not padded: the adaptive-routing hypothesis requires testing whether
equal accuracy can be achieved with less real compute.

The lock also fingerprints all executable `brain` Python sources and
`pyproject.toml` (excluding tests, generated results, docs, and bytecode). Thus a
decision-path edit requires a new pre-outcome lock instead of silently retaining
the old protocol ID. H1–H4 effect-size gates are evaluated mechanically and
written to `threshold_decisions.json`; paired uncertainty remains separately
reported rather than becoming a post-outcome replacement rule.

## Why this model

The [official model catalog](https://developers.openai.com/api/docs/models)
lists GPT-5.6 as the current family. The cost-sensitive GPT-5.6 Luna page did not
provide a dated snapshot when this protocol was locked. The
[GPT-5.4 mini model page](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
does provide `gpt-5.4-mini-2026-03-17`. Phase 3 therefore prefers a reproducible,
reasonably capable, affordable snapshot over a mutable newer alias.

That choice narrows the conclusion: an observed or absent architecture effect is
specific to this model and protocol. It is not a model-independent law. Testing
GPT-5.6 Sol/Terra/Luna or another family is a later capacity-sweep protocol.

## Exact accounting

The adapter uses OpenAI's
[token counting endpoint](https://developers.openai.com/api/docs/guides/token-counting)
with the same instructions, input, response schema, model, and reasoning setting
used for inference. The full per-call output allowance must fit before a paid
request starts. Returned input usage must match preflight exactly; otherwise the
run aborts. Structured output follows OpenAI's
[JSON-schema response format](https://developers.openai.com/api/docs/guides/structured-outputs).

## Leakage and identity controls

- Correct answers and fault annotations remain in evaluator-owned `Task` objects.
- The model sees neither condition names nor dynamic internal state.
- Long event IDs are replaced with local aliases; provenance roots are hashed.
- The adapter receives no task answer, formal outcome, previous response ID, or
  cross-call hidden conversation.
- One adapter instance is shared across all conditions; each condition keeps only
  its architecturally declared memory.
- Condition order is deterministically shuffled inside every task block to reduce
  time/order bias.

## Execution gates

Offline preflight:

```bash
PYTHONPATH=agent-lab python -m brain.experiments.formal
```

A live run additionally needs the optional OpenAI dependency, `OPENAI_API_KEY`,
`--execute`, and the first 16 characters of the verified lock hash. It refuses a
non-empty output directory, performs no automatic inference retries, records
task-level failures, and preserves partial append-only artifacts on abort.

Do not run one condition, inspect it, and then tune the others. The first live use
of the blind suite must be the acknowledged all-condition matrix.

## Claim boundary

Even a successful run can support only an architecture effect for the frozen
model, tasks, and budget. It cannot by itself establish emergent roles,
individuality, personality, subjectivity, consciousness, or biological realism.
