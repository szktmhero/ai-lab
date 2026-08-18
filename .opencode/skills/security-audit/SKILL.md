---
name: security-audit
description: セキュリティ脆弱性の検出と分析に使用。SQLインジェクション、コマンドインジェクション、デシリアライゼーション攻撃などを検出する。
---

# Security Audit Skill

## 検出対象
1. SQLインジェクション
   - フォーマットストリングでのクエリ構築
   - パラメータ化クエリの未使用
   - ユーザー入力の直接埋め込み

2. コマンドインジェクション
   - shell=Trueでのサブプロセス実行
   - ユーザー入力のシェルコマンドへの組み込み

3. デシリアライゼーション脆弱性
   - pickle.loads()の使用
   - 安全でないデシリアライゼーション

4. テンプレートインジェクション
   - ユーザー入力の直接テンプレート埋め込み
   - render_template_stringの不正使用

## 分析手順
1. ファイルを読み、インポート文を確認
2. リクエストハンドラを特定
3. ユーザー入力の流れを追跡
4. 危険なパターンを検出
5. 重大度を評価（Critical/High/Medium/Low）
6. 修正提案を生成

## 出力形式
```json
{
  "vulnerabilities": [
    {
      "type": "sql_injection",
      "severity": "critical",
      "location": "file:line",
      "description": "...",
      "fix": "..."
    }
  ]
}
```
