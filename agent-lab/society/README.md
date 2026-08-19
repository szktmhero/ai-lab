# 128-Agent Emergent Society Experiment

`lab_society.md` の最初のmilestoneを、structured actionsと差し替え可能な`AgentPolicy`で実装した実験環境です。

## Canonical Experiment

```text
128 agents
50 tasks
up to 20 rounds / task
3 matched seeds
```

比較条件:

- `independent`: social edgeなし。情報共有・新規contactなし
- `full`: 全agent間に初期edgeあり
- `local`: 平均degree 6の疎な初期graph。contactで変化可能

`experiments/results/independent/`, `full/`, `local/` がreview後の正式結果です。`random/` と `simple/` は初期実装時のlegacy生成物であり、比較には使用しません。

## Results

| Condition | Avg rounds | Avg proposals | Decision quality | Consensus | Communication cost | Information spread |
|-----------|------------|---------------|------------------|-----------|--------------------|--------------------|
| Independent | 5.0 | 128.0 | 0.793 | 0.008 | 640.0 | 0.173 |
| Full | 20.0 | 175.2 | 0.897 | 0.963 | 3113.1 | 0.330 |
| Local | 20.0 | 168.4 | 0.917 | 0.962 | 3391.2 | 0.320 |

`decision_quality` はsynthetic task内の潜在option品質を0-1へ正規化した値です。一般的な判断能力の尺度ではありません。

## Interpretation

**DESIGNED**

- communication budgetとaction cost
- network条件、trust更新式、情報分布
- evidenceを用いるSimplePolicyと最終decision score
- proposal mutation条件

**OBSERVED**

- Independentでは各agentの案が孤立し、平均consensusは0.008
- FullとLocalでは平均consensusが0.96前後
- FullとLocalはいずれも全taskで20 round上限に到達した
- Localの平均network densityは0.161、Fullは1.0
- Localの平均decision qualityは0.917
- proposal mutationは観測されたが、SimplePolicyではmergeは観測されなかった

**INFERRED**

- このsynthetic modelでは通信条件が情報利用と合意形成に影響した可能性がある
- seed単位の統計検定を行っていないため、条件差の一般化はできない

**SPECULATIVE**

- leadership、authority、coalition、specialization、institutionの創発は確認されていない
- 高いconsensusはSimplePolicyのsocial-evidence項によるherdingでも説明できる

## Outputs

各seed directoryには以下を保存します。

- `config.json`
- `events.jsonl`
- `rounds.json`
- `tasks.json`
- `agent_metrics.json`
- `society_state.json`
- `chronicle.md`
- `network.html`
- `timeline.html`
- `metrics.html`
- `proposals.html`

## Run

```bash
cd agent-lab/society
PYTHONPATH=.. .venv/bin/python -m pytest -q
PYTHONPATH=.. .venv/bin/python -c "from society.experiments.run import run_comparison; run_comparison()"
```
