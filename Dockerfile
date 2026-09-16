# Build multi-etapa. Decision de diseño explicita: el pipeline de
# investigacion completo usa 24.8M trades reales (~2GB, 10 dias x 2
# simbolos) -- impractico para un build de Docker rutinario. Esta imagen
# entrena sobre 1 dia real de BTCUSDT + ETHUSDT descargado en el build
# (BTCUSDT aporta 1.620.679 trades: 24 MB comprimidos, 140 MB de CSV),
# documentado aqui explicitamente en vez de escondido.
#
# La etapa final NO instala torch, jupyter ni optuna: la API solo sirve el
# modelo LightGBM, y arrastrar el stack de investigacion multiplicaria el
# tamaño de la imagen sin que nada lo use en runtime (ver requirements-api.txt).

FROM python:3.10-slim AS builder

ARG DATA_DATE=2026-08-25

# Parches de seguridad del sistema base: la etiqueta slim se publica con
# vulnerabilidades conocidas de paquetes Debian hasta su siguiente rebuild.
RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

# Descarga verificada por SHA256 e idempotente, la misma que se usa en local.
RUN python scripts/download.py --symbols BTCUSDT ETHUSDT --start "${DATA_DATE}" --end "${DATA_DATE}"

RUN python -m src.pipeline

FROM python:3.10-slim

RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY --from=builder /app/outputs/models/lightgbm_model.joblib outputs/models/lightgbm_model.joblib

# Usuario sin privilegios: el proceso no necesita escribir nada del proyecto.
RUN useradd --create-home --uid 10001 apiuser && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8000

# /health responde siempre; model_available distingue "el proceso vive" de
# "el proceso puede predecir".
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
