# Build multi-etapa. Decision de diseño explicita: el pipeline de
# investigacion completo usa 24.8M trades reales (~2GB, 10 dias x 2
# simbolos) -- impractico para un build de Docker rutinario. Esta imagen
# entrena sobre 1 dia real de BTCUSDT (~1.6M trades, ~24MB) en vez de los
# 10 dias x 2 simbolos completos: sigue siendo un modelo real entrenado con
# datos reales de Binance descargados en el build, solo que sobre un
# subconjunto, documentado aqui explicitamente en vez de escondido.

FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

RUN mkdir -p data/raw_btc data/raw_eth && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-08-25.zip', 'data/raw_btc/f.zip')" && \
    python -c "import zipfile; zipfile.ZipFile('data/raw_btc/f.zip').extractall('data/raw_btc/')" && \
    rm data/raw_btc/f.zip && \
    python -c "import urllib.request; urllib.request.urlretrieve('https://data.binance.vision/data/spot/daily/aggTrades/ETHUSDT/ETHUSDT-aggTrades-2026-08-25.zip', 'data/raw_eth/f.zip')" && \
    python -c "import zipfile; zipfile.ZipFile('data/raw_eth/f.zip').extractall('data/raw_eth/')" && \
    rm data/raw_eth/f.zip

RUN python -m src.pipeline

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY --from=builder /app/outputs/models/lightgbm_model.joblib outputs/models/lightgbm_model.joblib

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
