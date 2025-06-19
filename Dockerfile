# ---- Stage 1: Builder ----
FROM python:3.11-slim-bullseye AS builder
WORKDIR /install

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
 && pip install --prefix=/install_deps --no-cache-dir -r requirements.txt \
 && find /install_deps -type d -name "tests" -exec rm -rf {} + \
 && find /install_deps -type d -name "__pycache__" -exec rm -rf {} + \
 && rm -rf ~/.cache/pip

# ---- Stage 2: Runtime ----
FROM python:3.11-slim-bullseye AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install_deps /usr/local/
COPY app/ ./
COPY .env ./ 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
