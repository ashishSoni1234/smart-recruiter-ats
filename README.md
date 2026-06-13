# Intelligent Candidate Discovery & Ranking System

This repository contains the source code for the Redrob Hackathon (v4) candidate ranking challenge. The system is designed to parse the job description, extract core competencies, and rank 100,000 candidate profiles while strictly adhering to the 5-minute CPU compute constraint and offline-only requirements.

## 1. Setup Instructions

### Prerequisites
- Python 3.11+
- CPU with 16 GB RAM (No GPU required)

### Installation
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Reproduction Command

To reproduce the exact submission CSV from the raw candidate data, run the following command. The script handles the entire pipeline end-to-end within the 5-minute compute budget (averaging ~90 seconds on a standard 8-core CPU).

```bash
python pipeline.py --candidates data/candidates.jsonl --jd data/jd_extracted.txt --output solo_participant.csv --format jsonl
```

*Note: No pre-computation step is required. The pipeline runs from raw data to final CSV in a single step.*

## 3. System Architecture

The ranking system is built on a 5-stage funnel architecture designed to balance precision with strict compute constraints. 

### Stage 1: Sparse Pre-Filtering (`pipeline.py`)
To process 100,000 candidates within the time limit, running dense embeddings across the entire dataset is computationally infeasible. We implement an initial fast-pass using `BM25Okapi` on raw candidate text strings (summary, history, skills) against the job description anchor. This filters the pool down to the top 2,000 candidates in under 30 seconds.

### Stage 2: Heavy Enrichment & Verification (`enricher.py`)
For the filtered subset of 2,000 candidates, we apply our dense models:
- **Skill Verification Matrix:** Computes cosine similarity between claimed skills and actual career history descriptions using `all-MiniLM-L6-v2`. This acts as a defensive heuristic against keyword-stuffing.
- **Trust Score & Persona Extraction:** Derives a verified persona and trust score based on the veracity of the candidate's profile.

### Stage 3: Programmatic Disqualifiers (`scorer.py`)
Applies hard constraints to eliminate "trap" profiles:
- Disqualifies candidates with notice periods > 60 days.
- Eliminates profiles matching negative semantic anchors (e.g., pure researchers lacking production experience, or title-chasers).
- Computes a Behavioral Vibe score based on platform engagement signals.

### Stage 4: Hybrid Retrieval (`retrieval.py`)
Constructs a dual-index (Dense FAISS + Sparse BM25) for the remaining candidate pool. The system retrieves the top 300 candidates by evaluating similarity against the JD's Positive Anchor while penalizing similarity to the Negative Anchor.

### Stage 5: Cross-Encoder Reranking & Formatting (`pipeline.py`)
- **Reranking:** The top 300 pairs are passed through a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for precise semantic alignment.
- **Scoring:** The final score is a weighted blend of the Cross-Encoder output, Trust Score, and Behavioral Score.
- **Reasoning Generation:** Generates a dynamic, offline reasoning string based on the derived metrics to comply with the `has_network_during_ranking: false` rule.
- **Output:** The top 100 candidates are written to the CSV, with tie-breakers deterministically handled by sorting `candidate_id` in ascending order.

## 4. Sandbox Link

A working sandbox environment for Stage 1 validation is documented in the `submission_metadata.yaml`. It accepts a small candidate sample and executes the exact pipeline architecture described above.
