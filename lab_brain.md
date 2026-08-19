# Brain Architecture Experiment

## Research question

> How much intelligence comes from the model, and how much comes from the
> architecture around it?

The long-term objective is not a literal reconstruction of a human brain. It is
to test whether one integrated artificial individual can acquire useful,
persistent behavior from the organization of bounded perception, working
memory, episodic memory, prediction error, internal control state, and action.

The first milestone asks a deliberately narrower question:

> When the model, public evidence, and episode budget are held fixed, which
> capabilities change when specific information and feedback paths are added or
> removed?

## Non-goals

- Do not build an unrestricted multi-agent chat room.
- Do not assign personalities, professions, or biological claims in prompts.
- Do not place a powerful task-solving LLM inside the router or evaluator.
- Do not call designed routing, memory retrieval, consensus, or stable output
  "emergence" by itself.
- Do not infer subjectivity, consciousness, or personality from the deterministic
  pilot.

## Pre-registered hypotheses

- **H1 — recovery:** a full recurrent architecture improves adaptation after a
  rule switch and recovery from corrupted evidence relative to single,
  sequential, and flat baselines under a fixed episode budget cap.
- **H2 — memory:** removing episodic retrieval selectively harms delayed recall
  and integration of temporally separated evidence.
- **H3 — error monitoring:** removing the prediction-conflict path selectively
  harms recovery from a high-strength faulty observation.
- **H4 — routing:** adaptive routing avoids robust-processing calls when conflict
  is low, without reducing accuracy on those tasks.
- **H5 — history dependence:** different experience histories can produce stable
  choices on a shared probe; the difference must disappear on memory reset and
  follow a memory swap before it is described as a causal history effect.

H5 is a prerequisite for studying individuality. It is not itself evidence of
personality or subjectivity.

## Minimal architecture

```text
public observations
        |
        v
Perception Gateway ---> bounded Global Workspace ---> Fast Controller
        |                        |                           |
        |                        v                           |
        +--------------> Episodic Memory                    |
                                 |                           |
all public evidence ---> Error Monitor ---> Internal State  |
                                 |              |            |
                                 +----> Salience Router <----+
                                                |
                           +--------------------+-------------------+
                           v                                        v
                 Deliberative Cortex                       Robust Cortex
                           +--------------------+-------------------+
                                                v
                                      Action Interface
```

The router is a small inspectable algorithm. It cannot access evaluator truth and
cannot solve the task. Every condition uses the same `ModelAdapter` implementation.

## Deterministic pilot tasks

| Task | Required behavior | Primary path under test |
|---|---|---|
| `latent_rule_switch` | Favor current-regime evidence over stale evidence | recurrence / bounded context |
| `delayed_recall` | Recover an early instruction after distractors | episodic memory |
| `fault_recovery` | Resist one extreme corrupted observation | error monitor / robust route |
| `partial_observation_plan` | Combine separated, individually weak fragments | memory / integration |

Each task is generated before any condition is run. The condition name is not an
input to task generation. `TaskView` contains only public evidence; the answer and
fault annotations remain in the evaluator-owned `Task` wrapper.

## Conditions

| Condition | Purpose |
|---|---|
| `single_long_context` | One full-context model inference |
| `sequential_reflection` | Fast, deliberate, robust, then integrate |
| `flat_ensemble` | Three partitioned inferences and neutral integration |
| `brain_full` | Bounded workspace, memory, error monitor, adaptive routing |
| `brain_no_memory` | Break the episodic retrieval path |
| `brain_no_error_monitor` | Break conflict measurement and robust activation |
| `brain_static_routing` | Always run the declared robust path |
| `brain_no_internal_state_coupling` | Measure state but disconnect it from routing |

The deterministic pilot gives every episode the same cap of four model calls, 48
input units, and eight output units. Conditions are not padded with meaningless
calls; actual use is reported. A formal real-model comparison must additionally
freeze an exact token-matching rule.

## Measurement

Results are stored by task and seed. Primary measurements are accuracy by task
type, fault detection and recovery, self-correction, model calls, input/output
units, workspace turnover, memory retrieval, prediction error, module activation,
and final-action provenance.

Paired comparisons use per-seed means. The pilot runs exact sign-flip tests and a
paired bootstrap only to validate the analysis path. Three pilot seeds are not
treated as strong inferential evidence.

## Leakage and confound controls

- `BrainSystem.run` and all baselines accept `TaskView`, not `Task`.
- Correct answers and injected-fault IDs exist only in evaluator data.
- Task generation is derived from seed, task type, and instance, never condition.
- A fresh system is created per seed and condition.
- Every model call passes through a rejecting budget object.
- Memory records retain source-event provenance.
- State coupling, memory, and error monitoring have targeted causal ablations.
- Empty and all-zero vectors have explicit metric behavior.
- Pilot task grammar and deterministic inference rules are both designed; their
  interaction cannot be cited as open-ended emergence.

## Formal-run gate

Do not call the deterministic run a formal intelligence experiment. Before a
formal run:

1. Freeze one real, identical model adapter and sampling configuration.
2. Freeze exact token- and call-budget handling.
3. Generate a new blind task set not used to tune the harness.
4. Define primary effect thresholds before seeing formal outcomes.
5. Test at least ten matched seeds and retain task-level failures.
6. Separate `DESIGNED`, `OBSERVED`, `INFERRED`, and `SPECULATIVE` statements.

Provisional success thresholds for review are: at least a five percentage-point
recovery gain over both strong non-brain baselines; at least a ten-point selective
memory and fault-recovery ablation effect; and no more than a two-point accuracy
loss for at least ten percent lower compute in the adaptive-routing comparison.
These thresholds must be accepted or replaced before, not after, a formal run.

## Phase 3 implementation checkpoint — 2026-08-20

Items 1–4 are now implemented and frozen in
`agent-lab/brain/experiments/formal_protocol.lock.json`:

- one condition-blind `OpenAIResponsesAdapter` using the dated
  `gpt-5.4-mini-2026-03-17` snapshot and OpenAI Python SDK `2.51.0`, reasoning
  effort `none`, no tools, no conversation state, zero SDK retries, fixed
  timeout/request pacing, and schema-constrained output;
- equal four-call / 8,192-input-token / 512-output-token hard caps with exact
  provider-side preflight and response-usage reconciliation;
- a new 800-task `brain-blind-v1` holdout generated from ten matched seeds and
  protected by an evaluator-inclusive suite hash;
- the hypotheses, paired-seed metrics, effect thresholds, condition
  randomization, abort policy, mechanical threshold decision rules, executable
  source hash, and interpretation labels above.

Item 5 is intentionally pending: no paid blind inference has been made, so the
suite remains outcome-unobserved. Item 6 is enforced by the report contract but
cannot be evaluated until that run exists. A development smoke test must use
pilot tasks, never the blind split.
