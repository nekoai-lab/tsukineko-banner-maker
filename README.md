# 🐱 月猫バナーメーカー (tsukineko-banner-maker)

AIパワードのバナー生成ツール。OpenAI の画像生成APIを使用して、テンプレートベースで簡単にバナーを作成できます。

## ✨ 機能

- **12種類のテンプレート**: Generate系8個 + Edit系4個
- **バリエーション生成**: 色・構図・季節などの軸で複数バリエーションを一括生成
- **参照画像アップロード**: キャラ/色/素材の参照画像でスタイルを指定
- **コスト管理**: Final品質の生成回数制限と警告表示
- **プログレス表示**: 複数枚生成時のリアルタイム進捗表示

## 🚀 クイックスタート

### ローカル実行

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# 環境変数の設定
cp .env.example .env
# .env ファイルを編集して OPENAI_API_KEY を設定

# アプリ起動
streamlit run app.py
```

ブラウザで http://localhost:8501 にアクセス

### Docker実行

```bash
# イメージのビルド
docker build -t tsukineko-banner-maker .

# コンテナの起動
docker run -p 8080:8080 -e OPENAI_API_KEY=sk-proj-xxx tsukineko-banner-maker
```

ブラウザで http://localhost:8080 にアクセス

## ☁️ Cloud Run デプロイ

### 1. Google Cloud SDK のセットアップ

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Artifact Registry の作成（初回のみ）

```bash
gcloud artifacts repositories create tsukineko-repo \
    --repository-format=docker \
    --location=asia-northeast1 \
    --description="Tsukineko Banner Maker"
```

### 3. イメージのビルドとプッシュ

```bash
# Cloud Build でビルド
gcloud builds submit --tag asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/tsukineko-repo/tsukineko-banner-maker

# または、ローカルでビルドしてプッシュ
docker build -t asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/tsukineko-repo/tsukineko-banner-maker .
docker push asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/tsukineko-repo/tsukineko-banner-maker
```

### 4. Cloud Run へデプロイ

```bash
gcloud run deploy tsukineko-banner-maker \
    --image asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/tsukineko-repo/tsukineko-banner-maker \
    --platform managed \
    --region asia-northeast1 \
    --allow-unauthenticated \
    --set-env-vars OPENAI_API_KEY=sk-proj-xxx \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300
```

### 5. Secret Manager を使用（推奨）

```bash
# シークレットの作成
echo -n "sk-proj-xxx" | gcloud secrets create openai-api-key --data-file=-

# Cloud Run にシークレットをマウント
gcloud run deploy tsukineko-banner-maker \
    --image asia-northeast1-docker.pkg.dev/YOUR_PROJECT_ID/tsukineko-repo/tsukineko-banner-maker \
    --platform managed \
    --region asia-northeast1 \
    --allow-unauthenticated \
    --set-secrets OPENAI_API_KEY=openai-api-key:latest \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300
```

## 📁 プロジェクト構成

```
tsukineko-banner-maker/
├── app.py              # Streamlit メインアプリ
├── prompt_builder.py   # プロンプト生成ロジック
├── templates.yaml      # テンプレート定義
├── requirements.txt    # Python 依存パッケージ
├── Dockerfile          # Cloud Run 用 Dockerfile
├── .env.example        # 環境変数のサンプル
├── .dockerignore       # Docker ビルド除外設定
├── .gitignore          # Git 除外設定
└── README.md           # このファイル
```

## 🎨 テンプレート一覧

### Generate系（新規生成）

| ID | テンプレート名 | 生成枚数 | 説明 |
|----|---------------|---------|------|
| t01 | キャラ統一バナー | 3枚 | 構図違いの3パターン |
| t02 | 文字入りバナー | 1枚 | テキスト完全一致描画 |
| t03 | 季節感バリエーション | 4枚 | 春夏秋冬の4パターン |
| t04 | 色違いバナー | 3枚 | 配色バリエーション |
| t05 | サイズ展開テンプレ | 3枚 | SNS用サイズセット |
| t06 | シンプルロゴバナー | 1枚 | 余白多めミニマル |
| t07 | 情報詰め込み型 | 1枚 | イベント告知向け |
| t08 | ミニマルデザイン | 1枚 | 要素最小限 |

### Edit系（画像編集）

| ID | テンプレート名 | 編集対象 | 説明 |
|----|---------------|---------|------|
| t09 | 色のみ変更 | color | 配色だけを変更 |
| t10 | テキストのみ変更 | text | テキスト部分を差し替え |
| t11 | 背景のみ変更 | background | 背景を差し替え |
| t12 | 小物追加 | add_element | 装飾を追加 |

## ⚙️ 環境変数

| 変数名 | 必須 | 説明 |
|--------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI API キー |
| `ENV` | - | 環境識別子（development/production） |

## 📝 ライセンス

MIT License
