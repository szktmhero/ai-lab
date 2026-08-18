---
name: perf-audit
description: パフォーマンス問題の検出と分析に使用。N+1クエリ、メモリリーク、無駄なループなどを検出する。
---

# Performance Audit Skill

## 検出対象
1. N+1クエリ問題
   - ループ内でのDBクエリ実行
   - 一括取得の未活用

2. メモリリーク
   - 無制限なキャッシュ成長
   - 参照カウンタの問題
   - 大規模データの保持

3. 計算量の問題
   - O(n²)以上のアルゴリズム
   - 不要な繰り返し処理

4. I/Oボトルネック
   - 同期的な外部呼び出し
   - 並列処理の未活用

## 分析手順
1. データベースクエリパターンを特定
2. ループ構造を分析
3. メモリ使用パターンを追跡
4. 外部I/O呼び出しを特定
5. 重大度を評価
6. 最適化提案を生成

## 出力形式
```json
{
  "issues": [
    {
      "type": "n_plus_one_query",
      "severity": "high",
      "location": "file:line",
      "description": "...",
      "optimization": "...",
      "expected_improvement": "..."
    }
  ]
}
```
