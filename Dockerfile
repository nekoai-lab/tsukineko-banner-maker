# Dockerfile for tsukineko-banner-maker
# Cloud Run deployment

FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# システム依存パッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python依存パッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# 非rootユーザーの作成（セキュリティ）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Streamlit設定
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Cloud Run標準ポート
EXPOSE 8080

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# アプリケーション起動
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
