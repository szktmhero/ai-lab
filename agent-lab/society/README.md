# 128-Agent Emergent Society Experiment

`lab_society.md` の最初のmilestoneを、structured actionsと差し替え可能な
`AgentPolicy`で実装した実験環境です。

## Research Question

> Can a decision-making structure emerge from many limited autonomous agents
> without designing the structure itself?

現段階では、社会構造の創発を主張する前に、通信networkの効果とpolicyに明示した
同調ルールの効果を分離しています。

## Canonical Experiment

```text
128 agents
50 persistent tasks / seed
up to 20 rounds / task
10 matched seeds
```

5条件を比較します。

| Condition | Network | Support-count weight |
|-----------|---------|----------------------|
| `independent` | edgeなし | 0.00 |
| `full_evidence` | 全結合 | 0.00 |
| `local_evidence` | 初期平均degree 6 | 0.00 |
| `full_social` | 全結合 | 0.05 |
| `local_social` | 初期平均degree 6 | 0.05 |

乱数はagent属性、network、情報配布、simulation dynamicsで独立streamを使用します。
同じseedでは、条件間でagent属性、task、初期情報の正誤と配布先が一致します。

## Results

| Condition | Rounds | Limit rate | Proposals | Quality | Consensus | Comm. cost | Info spread |
|-----------|--------|------------|-----------|---------|-----------|------------|-------------|
| independent | 5.0 | 0.000 | 128.0 | 0.755 | 0.008 | 640.0 | 0.176 |
| full_evidence | 20.0 | 0.998 | 134.4 | 0.908 | 0.167 | 2885.0 | 0.337 |
| local_evidence | 20.0 | 0.996 | 130.4 | 0.905 | 0.094 | 3182.0 | 0.329 |
| full_social | 20.0 | 1.000 | 176.0 | 0.901 | 0.962 | 3111.5 | 0.335 |
| local_social | 20.0 | 0.998 | 166.9 | 0.902 | 0.962 | 3391.2 | 0.327 |

`decision_quality` はsynthetic task内の潜在option品質を0–1へ正規化した値であり、
一般的な判断力や知性の尺度ではありません。

## What the Ablation Shows

seedごとの50-task系列を1つの独立単位として、対応のあるsign-flip testを行いました。

- support-count項を無効にしても、networkあり条件はindependentよりqualityが約0.15高い
  (`p=0.002`)
- support-count項を追加するとconsensusはFullで`+0.795`、Localで`+0.867`
  (`p=0.002`)
- 同じ追加によるquality差はFullで`-0.007` (`p=0.252`)、Localで`-0.003`
  (`p=0.682`)
- support-count項なしではLocalとFullのquality差は`-0.002` (`p=0.770`)
- networkあり条件の99.6%以上が20 round上限に到達した

したがって、この実験の高consensusは社会構造の創発よりも、SimplePolicyへ明示した
support-count追随項でほぼ説明できます。一方、synthetic taskにおけるquality改善は、
その追随項ではなくnetworkを介した情報・proposal交換と整合します。ただし、下記の
independent baseline制約があるため、情報共有だけの因果効果とは解釈できません。

## Interpretation Boundaries

**DESIGNED**

- communication budgetとaction cost
- network topologyと情報分布
- optionalなsupport-count項
- trust更新式と最終decision score

**OBSERVED**

- 条件ごとのdescriptive metrics
- seed blockによる対応差、bootstrap interval、exact sign-flip p-value
- proposal mutation、support history、network変化、round上限到達

**NOT TESTED**

- trustはtask間で更新・保持されるが、SimplePolicyは情報評価や通信相手の選択に
  trustを利用しない。現在の実験にはtrust学習が判断へ戻る因果経路がない
- 情報発信者にtask間で安定したreliabilityがないため、専門性を学習できる環境にも
  まだなっていない
- independentでは128個のproposalが各1票のまま分断され、最終tie-breakはoption単位で
  private choiceを集約しない。そのためnetworkとの差は純粋な情報共有効果ではない
- leadership、authority、coalition、specialization、institutionの創発

## Outputs

正式比較として追跡する軽量成果物:

- `experiments/results/design.json`
- `experiments/results/comparison.json`
- `experiments/results/seed_metrics.json`
- `experiments/results/paired_statistics.json`
- `experiments/results/COMPARISON.md`

`save_artifacts=True`では、条件・seedごとにevent log、round/task metrics、agent metrics、
最終state、chronicle、HTML visualizationも生成します。これらは再生成可能なため
`.gitignore`対象です。

`experiments/results/random/` と `simple/` は修正前のlegacy生成物であり、正式比較には
使用しません。

## Run

```bash
cd agent-lab/society
uv venv
uv pip install numpy matplotlib pytest
PYTHONPATH=.. .venv/bin/python -m pytest -q
PYTHONPATH=.. .venv/bin/python -c "from society.experiments.run import run_comparison; run_comparison(save_artifacts=False)"
```

Linux等で条件を並列実行する場合は、適切な`__main__` guardを持つscriptから
`run_comparison(parallel=True, max_workers=4)`を呼び出します。

## Next Experiment

次は、連続値のsource reliability / category expertiseとprovenanceを環境側に導入し、
trustを情報評価・共有先選択へ接続した上で、`trust learning on/off`をablationします。
役割名は事前付与せず、反復task後の影響力・情報精度・network位置をpost-hocに測定します。
