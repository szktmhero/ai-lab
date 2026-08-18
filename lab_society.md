# 128-agent emergent society experiment

`agent-lab` に、128個の独立した主体の相互作用から、信頼・専門性・連合・権威・合意形成などの社会的構造が自発的に形成されるかを観察する実験環境を実装してください。

ディレクトリは以下とします。

```text
agent-lab/
  society/
```

この実験の目的は、128人のLLMに「議会ごっこ」をさせることではありません。

問いは次です。

> Can a decision-making structure emerge from many limited autonomous agents without designing the structure itself?

より具体的には、

> 中央司令塔も議会制度も固定された役職も持たない128個の主体が、情報・意見・信頼を局所的に交換し続けたとき、集団として意思決定するための構造が自発的に形成されるか？

を観察します。

---

# 1. Core principles

以下を重要な設計原則としてください。

* 中央意思決定者を作らない
* chairman / leader / moderator を最初から設定しない
* 多数決をデフォルトの意思決定方式にしない
* 全エージェントに同じ情報を与えない
* 全エージェントが全員と通信できる構造にしない
* 固定された専門家役を最初から設定しない
* 「派閥を作れ」「専門化しろ」「リーダーを選べ」と指示しない
* 社会制度をハードコードしない
* agent同士の相互作用から構造が生まれる余地を残す
* 同じ128 agentを複数の課題に継続参加させる
* agentの履歴・信頼関係・内部状態をtask間で維持する

重要なのは、

```text
128 agents
    ↓
local interaction
    ↓
repeated decisions
    ↓
?
```

の `?` を観察することです。

---

# 2. Initial population

128 agentsを生成してください。

各agentは最低限、次の状態を持ちます。

```text
id

preferences[]
beliefs[]
confidence

energy / communication_budget

memory

trust[other_agent]
known_agents[]

current_support
current_proposal
```

初期値には適度なrandomnessを入れてください。

ただし、

```text
Agent 01 = engineer
Agent 02 = economist
Agent 03 = leader
```

のような固定人格・固定役職を大量に人間側から設定することは禁止します。

初期差異は、

* preference vector
* risk tolerance
* novelty preference
* confidence
* communication tendency
* initial social connections

などの連続値として与えてください。

人格そのものを人間が設計しすぎないことを重視します。

random seedを指定可能にしてください。

---

# 3. Social network

agentは128人全員と直接通信できてはいけません。

初期状態では、各agentがランダムな少数のagentのみを知っているsocial graphを作ります。

例えば平均degreeは、

```text
4〜8
```

程度から始めてください。

agentはinteractionを通じて、

```text
follow
unfollow
trust
distrust
```

のような関係変更を行えるようにしてください。

その結果、

* hub
* cluster
* coalition
* isolation
* bridge agent

などが自然発生するか観察します。

---

# 4. Information asymmetry

各taskについて、全員に同じ情報を与えてはいけません。

例えば、

```text
Task:
今週末、128人の集団としてどこへ行くべきか？
```

という課題なら、

Agent A:

```text
天気予報の一部を知っている
```

Agent B:

```text
交通時間を知っている
```

Agent C:

```text
混雑情報を知っている
```

Agent D:

```text
候補地Aについて詳しい
```

Agent E:

```text
ほとんど情報を持っていない
```

というように情報を分散させます。

重要なのは、

> 良い集団判断を行うには、情報共有が必要

な環境を作ることです。

ただし、どの情報源が信頼できるかはagentに最初から教えません。

---

# 5. Agent actions

各roundでagentが行える行動を限定してください。

例えば、

```text
PROPOSE
SUPPORT
OPPOSE
MODIFY
MERGE
SHARE_INFORMATION
REQUEST_INFORMATION
CONTACT
FOLLOW
UNFOLLOW
WAIT
```

程度とします。

agentが直接長文ディベートを延々と続ける設計にはしないでください。

各actionには構造化されたpayloadを持たせます。

例:

```json
{
  "action": "SUPPORT",
  "proposal_id": 14,
  "confidence": 0.71
}
```

情報共有なら、

```json
{
  "action": "SHARE_INFORMATION",
  "target_agent": 82,
  "information_id": "weather_03"
}
```

proposal変更なら、

```json
{
  "action": "MODIFY",
  "proposal_id": 14,
  "change": "move outdoor activity to evening"
}
```

とします。

自然言語を完全禁止する必要はありません。

ただし、agent間通信の中心はstructured actionとし、無制限な会話を避けてください。

---

# 6. Communication cost

コミュニケーションは無料にしないでください。

各agentには1 taskにつき限られたcommunication budgetを与えます。

例えば、

```text
100 units
```

とし、

```text
support             1
oppose              1
share_information   3
request_information 2
create_proposal      5
contact_new_agent    5
```

のようにコストを持たせます。

正確な値は実験設定として変更可能にしてください。

目的は、

> 全員が全員と無限に喋れば解決する

状態を防ぐことです。

情報共有・探索・説得には資源が必要という環境を作ります。

---

# 7. Trust

agent間にtrustを持たせてください。

初期trustは小さいrandom valueとします。

Agent AがAgent Bから受け取った情報が後に正しかった場合、

```text
trust[A][B] ↑
```

誤っていた場合、

```text
trust[A][B] ↓
```

とします。

ただしtrust更新式は単純なものにし、複雑な社会行動をハードコードしないでください。

taskを跨いでtrustを維持します。

これにより、

```text
reliable information source
influencer
isolated agent
trusted cluster
```

などが自然に形成されるか観察します。

---

# 8. Proposal evolution

proposalは単なる候補ではなく、変異可能なオブジェクトとして扱ってください。

例えば、

```text
Proposal #7
浅草で昼飲み
```

が、

```text
Proposal #14
浅草 → 谷中 → 上野で夕方飲み
```

へ変化したり、

複数proposalが、

```text
MERGE
```

されるようにします。

記録するべきもの：

```text
proposal parent
mutation history
support history
creator
modifier
merge history
lifetime
```

これによって、

> agentではなくideaそのものが進化する

現象も観察したいです。

---

# 9. Decision

最終決定アルゴリズムを「単純多数決」に固定しないでください。

一定round経過後に、

* support distribution
* confidence
* proposal stability
* unresolved opposition
* network diffusion

などから「社会として最も定着したproposal」を選択する簡単なmechanismを用意してください。

ただし実装上必要なら、baselineとして多数決も用意して構いません。

重要なのは、

```text
EmergentDecision
vs
MajorityVote
```

を比較できることです。

---

# 10. Persistent society

このexperimentでは、1問解いて終わりにしないでください。

同じ128 agentsに、連続して複数taskを与えます。

例えば最初の50 tasksは軽い問題で構いません。

例：

```text
今週末どこへ行く？
今日何を食べる？
旅行先はどこにする？
予算10万円を何に使う？
どの映画を見る？
どの企画を採用する？
```

その後、

```text
簡単な設計問題
情報探索問題
不確実性のある判断
複数目的の最適化問題
```

などに広げられるようにしてください。

各task終了後も、

```text
trust
social graph
memory
behavioral state
```

を維持します。

目的は、

> 判断を繰り返すことで「社会」が育つか

を見ることです。

---

# 11. Emergent roles

役割は事前定義しません。

ただし後から分析できるよう、

agentごとに以下を計測してください。

```text
proposal creation count
proposal adoption rate

information accuracy
information propagation

number of incoming contacts
number of outgoing contacts

centrality

support influence

trust received
trust given

successful modifications
bridge behavior
```

これらから、

```text
leader-like
expert-like
broker-like
explorer-like
critic-like
follower-like
```

な役割が結果として現れたかを分析します。

重要：

agentに、

```text
"You are a leader."
```

とは絶対に指示しないでください。

役割はobserver側が事後的に分類するものです。

---

# 12. Metrics

最低限、以下を記録してください。

## Decision metrics

```text
decision_quality
decision_time
consensus_level
minority_size
proposal_count
proposal_mutation_count
```

## Network metrics

```text
average_degree
network_density
clustering_coefficient
connected_components
centralization
```

## Trust metrics

```text
average_trust
trust_variance
trust_concentration
```

## Information metrics

```text
information_spread
information_accuracy
information_reach
information_loss
```

## Agent diversity

```text
opinion_entropy
preference_diversity
behavioral_diversity
```

---

# 13. Important emergent phenomena

次の現象が発生するかを観察します。

ただし発生するようにコードを書いてはいけません。

```text
leadership
specialization
coalitions
factions

experts
brokers
hubs

echo chambers
polarization

consensus
deadlock

institution-like behavior
```

特に、

> taskの種類によって自然に中心agentが変わるか

を観察したいです。

例えば結果として、

```text
food tasks
  → Agent 71 becomes influential

technical tasks
  → Agent 12 becomes influential

uncertain tasks
  → Agent 96 becomes a bridge
```

のような現象が出るなら非常に興味深いです。

---

# 14. Agent intelligence abstraction

simulation engineとagent intelligenceは完全に分離してください。

例えば、

```text
AgentPolicy
```

interfaceを作ります。

最低限、

```text
RandomPolicy
SimplePolicy
LLMPolicy
```

を想定します。

最初のexperimentでは、

```text
RandomPolicy
SimplePolicy
```

でもsimulation全体が動くようにしてください。

その後、

```text
OpenAI
Claude
local LLM
tiny model
```

を差し替えられる構造にしてください。

128 agentそれぞれが必ず別LLM instanceである必要はありません。

同一modelを異なるagent stateで利用する方式でも構いません。

---

# 15. Model heterogeneity

将来的に、

```text
128 agents
```

のmodel構成も変更できるようにしてください。

例えば、

```text
128 × same model
```

だけでなく、

```text
4 strong models
+
124 lightweight models
```

や、

```text
8 medium models
+
120 tiny models
```

などを実験できるようにします。

ただし、

> strong model = leader

とは設定しないでください。

強いモデルが結果として中心になるのか、ならないのかを観察したいです。

---

# 16. Observer

simulation内のagentとは別に、

```text
Observer
```

を用意してください。

Observerは意思決定には参加しません。

Observerの役割は、

* event収集
* metrics計算
* network分析
* visualization
* experiment report作成

だけです。

Observerがagentへ助言や介入をしてはいけません。

---

# 17. Social event log

すべてのinteractionをeventとして記録してください。

例えば：

```json
{
  "round": 7,
  "agent": 51,
  "action": "SHARE_INFORMATION",
  "target": 12,
  "information": "weather_03"
}
```

また、

```text
Round 7

Proposal #14
support 17 → 39

Agent 51 introduced new information.

Agents 12, 48 and 93 changed support.

Coalition C fragmented.

Agent 82 became a major information hub.
```

のようなhuman-readable timelineも生成できるようにしてください。

---

# 18. Minutes / Chronicle

simulation終了後に、

```text
society chronicle
```

を生成してください。

これは普通の国会議事録ではなく、

> この社会で何が起きたか

を記述するものです。

例えば、

```text
Task 18

Agent 51が新しい情報を導入した。

それまで最有力だったProposal 8への支持が急減。

Agent 82を中心とするネットワークがその情報を急速に拡散した。

以前からAgent 82へのtrustが高かった集団では、
他集団より3 round早く支持変更が起きた。

Proposal 14とProposal 21はRound 11で統合。

最終的にProposal 27が社会全体へ定着した。
```

という形式です。

---

# 19. Visualization

最低限以下を可視化してください。

## Social graph

```text
node = agent
edge = social connection
edge strength = trust
node size = influence
```

## Proposal evolution

proposalの、

```text
birth
mutation
merge
death
adoption
```

を追えるようにします。

## Timeline

各roundについて、

```text
proposal support
network change
information propagation
```

が確認できるようにしてください。

可能ならinteractive HTMLで出力してください。

---

# 20. Experiment 01

最初の実験は単純な課題で行います。

```text
128 agents
50 tasks
20 rounds/task
```

程度から開始してください。

最初のtask群は、

```text
weekend activity selection
restaurant selection
travel destination
simple resource allocation
```

など、正解が一意ではない軽い問題で構いません。

目的は回答品質そのものではありません。

観察対象は、

```text
trust network
social structure
proposal dynamics
role differentiation
```

です。

---

# 21. Baselines

最低限、以下を比較してください。

```text
A. Independent agents
   通信なし

B. Full communication
   全員が全員と通信可能

C. Local society
   限られた通信 + trust + persistent network
```

同じtask、同じseed群で比較します。

見るものは、

```text
decision quality
communication cost
decision speed
network formation
information utilization
```

です。

---

# 22. Critical requirement

面白い挙動を作り込まないでください。

このexperimentで避けたいのは、

```text
人間が社会構造を実装
↓
agentがその通りに動く
↓
「社会が創発した」と解釈
```

することです。

必ず、

```text
DESIGNED
OBSERVED
INFERRED
SPECULATIVE
```

を区別してください。

experiment reportでも、

```text
Observed:
Agent 82へのincoming edgeが増加した。

Inferred:
Agent 82がinformation brokerとして機能した可能性がある。

Speculative:
社会的権威の初期形成と解釈できるかもしれない。
```

のように分類してください。

---

# 23. Suggested repository structure

```text
agent-lab/
  society/

    simulation/
    agents/
    policies/

    network/
    trust/

    information/
    proposals/

    observer/
    metrics/
    visualization/

    experiments/
    reports/

    tests/
```

より単純な設計が適切なら変更して構いません。

過剰設計は避けてください。

---

# 24. First implementation process

いきなりコードを書き始めないでください。

最初に、

1. 最小社会モデル
2. agent state
3. allowed actions
4. communication model
5. trust update
6. proposal evolution
7. decision mechanism
8. observable metrics
9. 何が「設計」で何が「創発」なのか

を整理してください。

その設計を短く提示した後、Experiment 01を実装してください。

---

# 25. First milestone

最初のmilestoneではLLMを128体接続する必要はありません。

まず、

```text
128 autonomous agents
+
persistent state
+
limited communication
+
information asymmetry
+
trust
+
proposal evolution
+
observer
```

が動くsimulationを完成させてください。

その後、LLMPolicyを接続してください。

最終的には、

```text
128 small minds
       ↓
repeated interaction
       ↓
social structure
       ↓
collective decision
       ↓
?
```

という問いを追える実験環境にしてください。

このrepositoryの目的は、

> 128人に正しい答えを出させることではない。

> **128の主体を置いたとき、答えを出すための「社会」が自ら形成されるかを見ることである。**
