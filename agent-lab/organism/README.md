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

### Survival Rate

| Policy | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Random | 0.30 | 0.46 | 0 | 1 |
| RuleBased | 39.40 | 3.41 | 34 | 46 |

### Key Observations

**Observed:**
- RandomPolicy: 128セル中 0-1セルが生存（平均0.3）
- RuleBasedPolicy: 128セル中 34-46セルが生存（平均39.4）
- RuleBasedPolicy は RandomPolicy より **130倍以上** の生存率

**Inferred:**
- エネルギー消費と資源獲得のバランスが生存に重要
- 低エネルギー時の「資源に向かって移動」ルールが生存率を大幅に向上
- 高エネルギー時の信号発信が集団協調の初期形態の可能性

**Speculative:**
- 信号多様性（signal_diversity）の違いは空間的配置の違いと相関する可能性
- RuleBasedPolicy のクラスター形成は偶然ではなく、資源獲得戦略の副産物の可能性

## Metrics

- `alive_cells` — 生存セル数
- `average_energy` — 平均エネルギー
- `largest_cluster_size` — 最大クラスターサイズ
- `number_of_clusters` — クラスター数
- `signal_diversity` — 信号の多様性
- `spatial_entropy` — 空間エントロピー
- `resource_consumption` — 資源消費量

## Visualization

`experiments/results/*/visualization.html` をブラウザで開いて、step-by-step でシミュレーションを観察できます。

- 円の色: エネルギー（赤=低、緑=高）
- 外枠: 信号（HSL色で多様性を表現）
- 背景: 資源分布（明=豊富、暗=少ない）
