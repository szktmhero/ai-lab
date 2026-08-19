# Society Experiment: Comparison Report

## Design

10 matched seeds × 50 persistent tasks × 128 agents; up to 20 rounds per task.

The comparison crosses network reach (`independent`, `full`, `local`) with the SimplePolicy support-count term (`0.00` or `0.05`). Agent attributes, tasks, and initial information assignments are matched within each seed.

## Metrics Summary

| Condition | Rounds | Limit rate | Proposals | Quality | Consensus | Comm. cost | Info spread |
|-----------|--------|------------|-----------|---------|-----------|------------|-------------|
| independent | 5.0 | 0.000 | 128.0 | 0.755 | 0.008 | 640.0 | 0.176 |
| full_evidence | 20.0 | 0.998 | 134.4 | 0.908 | 0.167 | 2885.0 | 0.337 |
| local_evidence | 20.0 | 0.996 | 130.4 | 0.905 | 0.094 | 3182.0 | 0.329 |
| full_social | 20.0 | 1.000 | 176.0 | 0.901 | 0.962 | 3111.5 | 0.335 |
| local_social | 20.0 | 0.998 | 166.9 | 0.902 | 0.962 | 3391.2 | 0.327 |

## Paired Seed Contrasts

Differences are `left - right`. Intervals bootstrap the paired seed means; p-values are two-sided exact sign-flip tests over seed blocks.

| Contrast | Metric | Difference | 95% CI | p |
|----------|--------|------------|--------|---|
| full_evidence − independent | Decision quality | +0.153 | [+0.130, +0.175] | 0.0020 |
| full_evidence − independent | Consensus | +0.160 | [+0.157, +0.162] | 0.0020 |
| full_evidence − independent | Information spread | +0.160 | [+0.159, +0.162] | 0.0020 |
| full_evidence − independent | Round-limit rate | +0.998 | [+0.994, +1.000] | 0.0020 |
| local_evidence − independent | Decision quality | +0.150 | [+0.123, +0.175] | 0.0020 |
| local_evidence − independent | Consensus | +0.087 | [+0.084, +0.089] | 0.0020 |
| local_evidence − independent | Information spread | +0.153 | [+0.151, +0.154] | 0.0020 |
| local_evidence − independent | Round-limit rate | +0.996 | [+0.990, +1.000] | 0.0020 |
| full_social − full_evidence | Decision quality | -0.007 | [-0.017, +0.004] | 0.2520 |
| full_social − full_evidence | Consensus | +0.795 | [+0.793, +0.797] | 0.0020 |
| full_social − full_evidence | Information spread | -0.002 | [-0.003, -0.001] | 0.0020 |
| full_social − full_evidence | Round-limit rate | +0.002 | [+0.000, +0.006] | 1.0000 |
| local_social − local_evidence | Decision quality | -0.003 | [-0.018, +0.012] | 0.6816 |
| local_social − local_evidence | Consensus | +0.867 | [+0.864, +0.871] | 0.0020 |
| local_social − local_evidence | Information spread | -0.003 | [-0.003, -0.002] | 0.0020 |
| local_social − local_evidence | Round-limit rate | +0.002 | [-0.004, +0.008] | 1.0000 |
| local_evidence − full_evidence | Decision quality | -0.002 | [-0.017, +0.011] | 0.7695 |
| local_evidence − full_evidence | Consensus | -0.073 | [-0.076, -0.070] | 0.0020 |
| local_evidence − full_evidence | Information spread | -0.007 | [-0.008, -0.007] | 0.0020 |
| local_evidence − full_evidence | Round-limit rate | -0.002 | [-0.008, +0.004] | 1.0000 |

## Interpretation Boundaries

**DESIGNED:** communication limits, action costs, topology, the optional support-count term, trust updates, and the final decision score.

**OBSERVED:** the first table contains task-level descriptive means. The contrast table treats a complete persistent task sequence as one replicate.

**INFERRED:** a topology effect is separated from the explicit conformity term only by the named paired contrasts; p-values describe this simulator, not a population of real societies.

**NOT TESTED:** trust values persist and update, but SimplePolicy does not yet use trust to value evidence or choose communication partners. Therefore cross-task trust learning has no policy-level causal path in this experiment.

**BASELINE LIMITATION:** independent agents create separate proposals that remain tied at one supporter each. The final proposal-level tie break does not aggregate their private choices by option, so network-versus-independent quality differences are not a pure estimate of information sharing.

**SPECULATIVE:** leadership, coalitions, specialization, authority, and institutions are not established by this run.

`decision_quality` is normalized latent option quality in this synthetic task model; it is not a general measure of judgment or intelligence.