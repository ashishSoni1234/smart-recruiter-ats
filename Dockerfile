# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy dependency file first (layer caching) ─────────────────────────────
COPY requirements.txt .

# ── Install Python dependencies (CPU-only torch) ───────────────────────────
# Note: pip install needs internet — this is the PRE-COMPUTATION step.
# The RANKING step (python pipeline.py ...) needs NO internet.
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download HuggingFace models (so ranking step is fully offline) ──────
# This bakes the models into the image — no network needed at inference time.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Downloading cross-encoder...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models downloaded and cached.')"

# ── Copy full source code ───────────────────────────────────────────────────
COPY . .

# ── Validate that pre-computed JD cache exists ─────────────────────────────
RUN python -c "\
import os; \
assert os.path.exists('data/jd_intent_cache.json'), 'JD cache missing! Commit data/jd_intent_cache.json'; \
print('JD intent cache: OK')"

# ── Default command: run the ranking pipeline ──────────────────────────────
# Runtime: ~90 seconds on 8-core CPU | RAM: ~4-6 GB | No network required
CMD ["python", "pipeline.py", \
     "--candidates", "data/candidates.jsonl", \
     "--jd", "data/jd_extracted.txt", \
     "--output", "ashish-soni-solo.csv", \
     "--format", "jsonl"]
