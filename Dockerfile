FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock .python-version README.md ./
COPY home_credit_risk/ home_credit_risk/
COPY configs/ configs/
COPY main.py ./

RUN uv sync --no-dev
