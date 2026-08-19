# 128-Cell Emergent Organism Experiment

> How much intelligence comes from the model, and how much comes from the environment?

## Overview

128個の単純なセルが、局所的・非言語的な相互作用だけから、構造・役割分化・自己維持のような挙動を形成するかを観察する実験。

## Design Principles

- セル同士は自然言語で通信しない
- 中央司令塔を作らない
- セルは世界全体を観測できない（Moore neighborhood のみ）
- 「集団を作れ」などのゴールを直接与えない
- 生命らしい構造をハードコードしない

## Architecture

```
organism/
├── core/
│   ├── types.py          # Vec2, Action, 定数
│   ├── cell.py           # Cell dataclass
│   └── world.py          # World, simulation loop
├── policies/
│   ├── base.py           # CellPolicy ABC
│   ├── random.py         # RandomPolicy (baseline)
│   ├── rule_based.py     # RuleBasedPolicy
│   └── model.py          # ModelPolicy (stub)
├── metrics/
│   └── collector.py      # Metrics collection
├── visualization/
│   └── renderer.py       # HTML canvas visualization
├── experiments/
│   ├── run.py            # Single experiment runner
│   ├── batch_run.py      # Batch experiment runner
│   └── compare.py        # Comparison report generator
└── tests/
    └── test_core.py      # Unit tests
```

## Quick Start

```bash
cd agent-lab/organism
uv venv
uv pip install numpy
.venv/bin/python tests/test_core.py
.venv/bin/python experiments/batch_run.py
```

## Experiment Results (10 seeds × 1000 steps)

### Final Survivors

| Policy | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Random | 0.20 | 0.40 | 0 | 1 |
| RuleBased | 46.30 | 2.49 | 43 | 50 |

### Key Observations

**Observed:**
- RandomPolicy: 128セル中 0-1セルが生存（平均0.2）
- RuleBasedPolicy: 128セル中 43-50セルが生存（平均46.3）
- RuleBasedPolicy の最終生存数は平均で46.1セル多い
- RuleBasedPolicy の最終最大clusterは平均2.8セルで、大規模構造は観測されていない
- RuleBasedPolicy の最終signal diversityは0.993。ただしsignal規則にはnoiseと近傍平均が設計されている

**Inferred:**
- この環境では、低energy時のconsumeと隣接resource比較を含むpolicyが生存に寄与した
- `Survivors / 1000 Resource` は Random 0.026、RuleBased 1.212だった

**Speculative:**
- signalに共有された意味が発生したかは未検証
- 小規模clusterが自己組織化なのか、衝突とresource配置の副作用なのかは未検証
- 生命・知性・役割分化が生じたとは結論できない

## Metrics

- `alive_cells` — 生存セル数
- `average_energy` — 平均エネルギー
- `largest_cluster_size` — 最大クラスターサイズ
- `number_of_clusters` — クラスター数
- `signal_diversity` — 信号の多様性
- `spatial_entropy` — 空間エントロピー
- `resource_consumption` — 累積資源消費量
- `resource_remaining` — 現在の残存資源量
- `resource_regenerated` — 累積資源再生量

## Visualization

`experiments/results/*/visualization.html` をブラウザで開いて、step-by-step でシミュレーションを観察できます。

- 円の色: エネルギー（赤=低、緑=高）
- 外枠: 信号（HSL色で多様性を表現）
- 背景: 資源分布（明=豊富、暗=少ない）
