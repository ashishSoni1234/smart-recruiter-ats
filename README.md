# 🤖 Smart Recruiter ATS — AI Candidate Ranking System

> **Redrob Hackathon v4 | Data & AI Challenge**  
> Submitted by: Ashish Soni | Team: ashish-soni-solo

A 5-stage hybrid AI ranking pipeline that ranks candidates the way a great recruiter would — not by matching keywords, but by understanding who genuinely fits the role.

---

## 📋 Table of Contents
1. [Setup Instructions](#1-setup-instructions)
2. [Reproduction Command](#2-reproduction-command)
3. [System Architecture](#3-system-architecture)
4. [Pre-Computation Step](#4-pre-computation-step)
5. [Scoring Formula](#5-scoring-formula)
6. [Sandbox Demo](#6-sandbox-demo)

---

## 1. Setup Instructions

### Prerequisites
- Python 3.11+
- CPU with 16 GB RAM (No GPU required)

### Installation
```bash
# 1. Clone the repo
git clone https://github.com/ashishSoni1234/smart-recruiter-ats.git
cd smart-recruiter-ats

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 2. Reproduction Command

### Step A — Pre-Computation (One-time, needs internet)

Download the HuggingFace models to local cache (only needed once):

```bash
# Download and cache embedding model + cross-encoder (~150MB total)
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('Models cached.')"
```

The JD intent cache is already committed to the repo (`data/jd_intent_cache.json`) — no Groq API call needed.

### Step B — Ranking (Fully Offline)

To reproduce the **exact submission CSV** from raw candidate data:

```bash
python pipeline.py --candidates data/candidates.jsonl --jd data/jd_extracted.txt --output ashish-soni-solo.csv --format jsonl
```

**Expected runtime:** ~90 seconds on 8-core CPU  
**RAM usage:** ~4-6 GB peak  
**Network required:** ❌ No (models pre-cached, JD intent pre-cached)

### Step C — Validate Output

```bash
python data/validate_submission.py ashish-soni-solo.csv
```

### Docker (Alternative)

```bash
# Build image (downloads models during build — needs internet once)
docker build -t smart-recruiter-ats .

# Run ranking (fully offline — produces ashish-soni-solo.csv in container)
docker run --rm -v $(pwd)/data:/app/data smart-recruiter-ats
```

---

## 3. System Architecture

The pipeline is a **5-stage funnel** designed to balance precision with the 5-minute CPU constraint.

### Stage 1 — Sparse Pre-Filtering (`pipeline.py`)
**Input:** 100,000 candidates | **Output:** Top 2,000

Fast BM25 pre-filter on raw candidate text (summary + title + skills + career history descriptions). Reduces the pool from 100K to 2,000 in under 30 seconds without any dense model inference.

### Stage 2 — Heavy Enrichment & Verification (`enricher.py`)
**Input:** Top 2,000 | **Output:** Enriched metadata per candidate

For each of the top 2,000 candidates, we compute:
- **Skill Verification (Lie Detector):** Cosine similarity between claimed skills and actual career history using `BAAI/bge-small-en-v1.5`. Skills with <0.30 semantic similarity AND no keyword/partial match are flagged as potentially hallucinated.
- **True Persona Extraction:** Derives the candidate's real role (ML Engineer, Data Scientist, Product Manager, etc.) from career descriptions using embedding-based similarity to 11 standard role archetypes.
- **Honeypot Detection:** Catches candidates with (a) claimed experience far exceeding career history timeline, or (b) "expert" proficiency in 8+ skills with zero endorsements and zero duration.
- **Company DNA Analysis:** Distinguishes product companies vs. consulting firms (TCS, Infosys, Wipro, etc.).
- **Comprehensive Behavioral Score:** Uses all 23 Redrob platform signals — response rate, recency, interview completion rate, saved by recruiters, offer acceptance rate, profile completeness.
- **External Validation Score:** GitHub activity score + platform skill assessment scores.
- **Education Tier Score:** Maps `tier_1` through `tier_4` institution tiers to a 0-100 score.

### Stage 3 — Programmatic Disqualifiers (`scorer.py`)
**Hard DQ (score = 0):** Notice period >90 days, confirmed honeypot profiles  
**Soft penalties (score multiplier):** Ghost candidates (180d inactive + <20% response rate), high skill hallucination rate (>40%), notice period 30-90 days (gradual penalty)

### Stage 4 — Hybrid Retrieval (`retrieval.py`)
**Input:** Passed candidates | **Output:** Top 300

Constructs a dual-index (Dense FAISS + Sparse BM25) for the remaining pool. Retrieval uses **Reciprocal Rank Fusion (RRF, k=60)** to merge dense and sparse rankings. Crucially, similarity to the **Negative Semantic Anchor** (a crafted description of who NOT to hire) is subtracted from the positive similarity score, pushing keyword-stuffers and anti-profiles down.

### Stage 5 — Cross-Encoder Reranking & Scoring (`pipeline.py`)
**Input:** Top 300 | **Output:** Final ranked top 100

- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (JD anchor, candidate doc) pair with precise semantic alignment
- **Final Score Blend:** S1 (semantic, 50%) + S2 (culture/stability, 15%) + S3 (availability/behavioral, 12%) + S4 (trust/skill-verification, 10%) + S5 (GitHub/external, 8%) + S6 (education, 5%)
- **Reasoning:** Each candidate gets a unique, data-driven reasoning string referencing their actual verified skills, years of experience, persona, GitHub activity, trust score, notice period, and behavioral engagement

---

## 4. Pre-Computation Step

The JD parsing (Groq API call) is a one-time pre-computation step. The result is cached to `data/jd_intent_cache.json`.

**The ranking step reads only from the cache — zero network calls.**

To regenerate the JD intent cache (optional):
```bash
# Requires GROQ_API_KEY in .env file
python jd_parser.py
```

The cache is already committed to the repository. No pre-computation is needed for reproduction.

---

## 5. Scoring Formula

```
Final Score = base_score × soft_penalty_multiplier

base_score = (S1 × 0.50) + (S2 × 0.15) + (S3 × 0.12) + (S4 × 0.10) + (S5 × 0.08) + (S6 × 0.05)

S1 = (CrossEncoder_score × 0.65) + (HybridRRF_score × 0.35)   [Semantic match]
S2 = f(avg_tenure, company_DNA, title_chaser_check)             [Culture/stability]
S3 = blend(notice_period, recency, open_flag, behavioral_vibe)  [Availability]
S4 = verified_skills / total_skills                             [Trust]
S5 = f(github_activity, skill_assessment_scores, endorsements)  [External validation]
S6 = education_tier_mapping                                     [Education]
```

---

## 6. Sandbox Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashishSoni1234/smart-recruiter-ats/blob/main/sandbox_demo.ipynb)

The Colab notebook runs the full pipeline on `data/sample_candidates.json` (included in the repo) and validates the output format using the official `validate_submission.py` script.
