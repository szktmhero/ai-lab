import os
import pickle
import subprocess
import sqlite3
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# 脆弱性1: SQLインジェクション
@app.route('/user')
def get_user():
    user_id = request.args.get('id')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # 直接ユーザー入力を使用（危険）
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    user = cursor.fetchone()
    return jsonify(user)

# 脆弱性2: コマンドインジェクション
@app.route('/ping')
def ping():
    host = request.args.get('host')
    # 直接シェルコマンドを実行（危険）
    result = subprocess.run(f"ping -c 1 {host}", shell=True, capture_output=True, text=True)
    return result.stdout

# 脆弱性3: デシリアライゼーション攻撃
@app.route('/load', methods=['POST'])
def load_data():
    data = request.get_data()
    # 安全でないデシリアライゼーション（危険）
    obj = pickle.loads(data)
    return str(obj)

# 脆弱性4: テンプレートインジェクション
@app.route('/greet')
def greet():
    name = request.args.get('name')
    # ユーザー入力をテンプレートで直接使用（危険）
    template = f"Hello {name}!"
    return render_template_string(template)

# パフォーマンス問題1: N+1クエリ問題
@app.route('/posts')
def get_posts():
    conn = sqlite3.connect('blog.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts")
    posts = cursor.fetchall()
    result = []
    for post in posts:
        # 各投稿に対して別々のクエリを実行（非効率）
        cursor.execute(f"SELECT * FROM comments WHERE post_id = {post[0]}")
        comments = cursor.fetchall()
        result.append({"post": post, "comments": comments})
    return jsonify(result)

# パフォーマンス問題2: メモリリーク
cache = {}
@app.route('/cache/<key>')
def get_cached(key):
    value = request.args.get('value')
    if value:
        # キャッシュが無限に増加する（メモリリーク）
        cache[key] = value * 1000000
    return cache.get(key, "not found")

# アーキテクチャ問題: 関心事の分離が欠如
@app.route('/process', methods=['POST'])
def process_order():
    data = request.json
    
    # ビジネスロジック、データアクセス、プレゼンテーションが混在
    if not data.get('product_id'):
        return jsonify({"error": "product_id required"}), 400
    
    conn = sqlite3.connect('orders.db')
    cursor = conn.cursor()
    
    # 在庫確認
    cursor.execute(f"SELECT stock FROM products WHERE id = {data['product_id']}")
    stock = cursor.fetchone()
    
    if stock and stock[0] > 0:
        # 注文作成
        cursor.execute(
            f"INSERT INTO orders (product_id, user_id) VALUES ({data['product_id']}, {data['user_id']})"
        )
        conn.commit()
        
        # 在庫更新
        cursor.execute(
            f"UPDATE products SET stock = stock - 1 WHERE id = {data['product_id']}"
        )
        conn.commit()
        
        return jsonify({"status": "success"})
    else:
        return jsonify({"error": "out of stock"}), 400

# エラーハンドリングの欠如
@app.route('/divide')
def divide():
    a = int(request.args.get('a', 0))
    b = int(request.args.get('b', 1))
    # ゼロ除算エラーをハンドリングしていない
    return jsonify({"result": a / b})

if __name__ == '__main__':
    # 開発サーバーが本番で実行される（設定ミス）
    app.run(debug=True, host='0.0.0.0')
