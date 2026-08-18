# 128-cell emergent organism experiment

`agent-lab` に、128個の単純なセルの局所相互作用から自己組織化が発生するかを観察するための実験環境を実装してください。

この実験の目的は「生命らしいものをハードコードすること」ではありません。

問いは次です。

> How much intelligence comes from the model, and how much comes from the environment?

特に今回は、

> 複雑な中央制御を持たない128個のセルが、局所的・非言語的な相互作用だけから、構造・役割分化・自己維持のような挙動を形成するか？

を観察したいです。

## 基本原則

以下を必ず守ってください。

* セル同士は自然言語で通信しない
* 中央司令塔を作らない
* セルは世界全体を観測できない
* 各セルは近傍しか観測できない
* 「集団を作れ」「壁を作れ」「役割分担しろ」などのゴールを直接与えない
* 特定の生命らしい構造をハードコードしない
* ルールは可能な限り少なくする
* emergent behavior を観察できることを最優先する

## Experiment 01

128セルを2次元グリッド上にランダム配置します。

例：

* world: 32 x 32 grid
* cells: 128
* initial position: random
* simulation: 1000 steps
* random seed を指定可能にする

セルはそれぞれ独立した内部状態を持ちます。

最低限、以下を持たせてください。

```text
id
position
energy
age
internal_state[]
signal_output[]
```

`internal_state` と `signal_output` は固定長の数値ベクトルとしてください。

最初は例えば8〜16次元程度で構いません。

## Cell perception

各セルが観測できるのは自分自身と近傍だけです。

例えば Moore neighborhood を使用し、

```text
周囲8マス
+
自分自身の状態
+
近傍セルが出しているsignal
+
近傍環境のresource
```

だけを入力としてください。

グローバルなセル数、世界全体の状態、他の遠隔セルの情報などは与えません。

## Communication

セル同士の通信には自然言語を使用しません。

通信は数値ベクトルだけにしてください。

例：

```text
signal = [
  0.18,
 -0.72,
  0.03,
  0.91
]
```

signalの各次元に、人間側で意味を割り当てないでください。

つまり、

```text
signal[0] = danger
signal[1] = food
```

のような意味付けは禁止です。

意味が発生するのであれば、セル同士の相互作用から自然に形成されるようにしてください。

## Environment

環境にはresourceを存在させます。

resourceはセルがenergyを維持するために必要です。

ただしresource配置は完全に均一にはせず、空間的な偏りを作ってください。

例：

* 一部の場所ではresourceが豊富
* 一部では少ない
* 時間経過で多少変化する

セルはresourceを取得するとenergyが増えます。

存在するだけでもenergyを少しずつ消費します。

energy <= 0 でcellは死亡します。

## Cell actions

各stepでセルが選択できるactionは少数に限定します。

例えば：

```text
stay
move north
move south
move east
move west
consume resource
emit signal
```

必要であれば、

```text
attach
detach
```

を追加しても構いません。

ただし最初の実験では、分裂・繁殖・突然変異はまだ入れないでください。

まず128セル固定で、局所相互作用だけを観察します。

死亡したセルは復活させず、simulation中のcell数減少も観察対象とします。

## Cell policy

セルの意思決定ロジックを差し替え可能なinterfaceとして設計してください。

例えば、

```text
CellPolicy
```

のような抽象化を作ります。

最低限以下を実装してください。

### RandomPolicy

ランダム行動。

これはbaselineです。

### RuleBasedPolicy

非常に単純なルールベース。

複雑な生命的行動は組み込まないでください。

### ModelPolicy

将来的にtiny model / local LLM / neural networkを接続できるinterface。

Experiment 01ではModelPolicyの実モデル接続まで必須ではありません。

ただし後から、

* Ollama
* llama.cpp
* small transformer
* tiny neural network

などをセルのpolicyとして接続できる構造にしてください。

重要なのは、

**simulation engine と cell intelligence を分離することです。**

## Metrics

人間が結果を判断しやすいように、少なくとも以下を記録してください。

```text
step
alive_cells
average_energy
energy_variance

average_neighbor_count
largest_cluster_size
number_of_clusters

average_signal_magnitude
signal_variance

average_distance_between_cells
resource_consumption
```

可能なら以下も計測してください。

```text
spatial entropy
cluster entropy
cell-state diversity
signal diversity
```

「生命らしさ」のスコアを一つにまとめる必要はありません。

むしろ複数の指標をそのまま観察したいです。

## Visualization

simulationを人間が視覚的に観察できるようにしてください。

最低限、

* 2D grid
* cell position
* resource distribution
* cell energy

が確認できるようにします。

可能ならstepごとのanimationまたはGIF/HTML visualizationを生成してください。

さらにセルのsignalやinternal_stateからPCA等で2〜3次元へ圧縮して、色や表示に利用できると良いです。

ただしvisualizationのためにセルの内部動作を変更しないでください。

## Logging

各実験結果を保存してください。

例えば：

```text
experiments/
  exp-001/
    config.json
    metrics.csv
    final_state.json
    events.jsonl
    visualization.html
```

configには必ず、

```text
random_seed
world_size
cell_count
steps
policy
parameters
```

を残してください。

同一条件で再現できることを重要視します。

## Comparison

最初の実装が完成したら、最低でも以下を同じseed群で比較してください。

```text
RandomPolicy
vs
RuleBasedPolicy
```

seedは最低10種類。

各simulation 1000 steps。

そして、

* 生存率
* cluster形成
* signal diversity
* spatial entropy
* resource efficiency

に違いが出るか確認してください。

## Important

今回やりたいことは、

「賢いエージェントを128体作る」

ことではありません。

むしろ、

```text
simple cell
   +
simple cell
   +
simple cell
   ...
   ×128
        ↓
local interaction
        ↓
?
```

の `?` を観察することです。

そのため、設計時に人間側から高度な協調戦略を埋め込まないでください。

何か面白い挙動が発生した場合、

「その挙動がどのルールから生まれたのか」

を後から追跡できるよう、コードを単純かつinstrumentableにしてください。

## Repository structure

既存の `agent-lab` がある場合はその中に、

```text
organism/
```

を作ってください。

概ね以下のような責務分割を想定しています。

```text
agent-lab/
  organism/
    simulation/
    cell/
    policies/
    environment/
    metrics/
    visualization/
    experiments/
    tests/
```

ただし、より単純で適切な構成があるなら変更して構いません。

## Implementation policy

まずコードを書き始める前に、

1. 実験モデル
2. 最小ルール
3. architecture
4. metrics
5. 「どこまでが設計された挙動で、どこからが創発と言えるか」

を整理してください。

その後、Experiment 01を実装してください。

過剰設計を避け、まず動く小さなsimulationを完成させてください。

完成後、

1. testを実行
2. 10 seeds × RandomPolicy
3. 10 seeds × RuleBasedPolicy

を実行し、

観察された違いをREADMEまたはexperiment reportにまとめてください。

結果については、生命・知性・創発が発生したと安易に結論づけず、

* observed
* inferred
* speculative

を明確に区別してください。
