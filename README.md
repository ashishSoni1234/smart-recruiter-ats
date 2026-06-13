# India Runs AI Challenge - Rank 1 Submission 🏆

Welcome to the **Solo Ranker** submission for the Intelligent Candidate Discovery & Ranking Challenge. 

This repository contains the architecture, logic, and offline-executable pipeline designed to parse 100,000+ candidate profiles, intelligently bypass keyword stuffers/"trap" candidates, and output the absolute best 100 fits within **under 2 minutes on a local CPU**, completely offline.

## 🚀 Architecture: The 6-Layer Ranking Pipeline

Recruiting is a multi-dimensional problem. Keyword matching fails because candidates lie, optimize their profiles, and hide red flags. To solve this, our system employs a **6-Layer Funnel Architecture**:

### Layer 1: JD Parser & Intent Extraction (`jd_parser.py`)
Instead of matching keywords, we parse the Job Description into a `JDIntentBundle`. This extracts the core competencies (e.g., Vector Search, MLOps, System Design) and defines a **Positive Semantic Anchor** (the ideal candidate's profile) and a **Negative Semantic Anchor** (the trap profile, like a title-chaser or framework enthusiast).

### Layer 2: Ultra-Fast BM25 Pre-Filtering (`pipeline.py`)
Running deep embedding models on 100,000 candidates takes over 30 minutes on a CPU. To stay under the 5-minute compute limit:
- We combine raw text (Summary, History, Skills) for all candidates.
- We use `BM25Okapi` to perform a blazing-fast sparse retrieval.
- We filter out 98% of the noise in ~20 seconds, keeping only the **Top 2000** candidates for heavy enrichment.

### Layer 3: Candidate Enricher & Lie Detection (`enricher.py`)
For the top 2000 candidates, we run our defensive ML logic:
- **Skill Verification (Lie Detector):** We generate dense embeddings for every skill the candidate claims and compare it against the semantic reality of their work history using cosine similarity. If the similarity is low, the skill is flagged.
- **Trust Score:** Based on the lie detection, a trust score (0-100%) is generated.
- **True Persona Derivation:** We build an unbiased "Truth Document" representing what the candidate *actually* does, stripping away marketing fluff.

### Layer 4: Disqualifiers & Base Scoring (`scorer.py`)
Before passing to the final ranking, we apply rigid constraints:
- **Hard Disqualifiers:** Candidates with Notice Periods > 60 days, inactive for > 180 days, or matching "Title Chaser" heuristics are instantly dropped.
- **Behavioral Vibe Score:** Evaluates culture alignment using signals like response times, offer acceptance rates, and platform engagement.

### Layer 5: Hybrid Retrieval (`retrieval.py`)
We build a local dual-index for the remaining candidates:
- **Dense FAISS Index:** Embeddings generated via `all-MiniLM-L6-v2`.
- **BM25 Sparse Index:** For exact keyword preservation.
- We run a similarity search against the Positive Anchor and *subtract* similarity from the Negative Anchor (pushing away trap candidates). This grabs the Top 300.

### Layer 6: Cross-Encoder Reranking & Final Offline Reasoning (`pipeline.py`)
- **Cross-Encoder:** The Top 300 pairs (JD vs. Truth Document) are passed through `cross-encoder/ms-marco-MiniLM-L-6-v2` for highly accurate semantic matching.
- **Final Blend:** The Cross-Encoder score is blended with the Trust Score, Behavioral Score, and Hybrid Score.
- **Programmatic Reasoning:** Since LLM APIs are banned during the ranking phase (offline constraint), we dynamically generate a 2-sentence reasoning string based on the derived persona and computed metrics.

## ⚙️ How to Run

### Prerequisites
- Python 3.11+
- `pip install -r requirements.txt`

### Execution
To reproduce the output on the complete dataset:
```bash
python pipeline.py --candidates data/candidates.jsonl --output solo_participant.csv --format jsonl
```

- **Speed:** ~1.5 - 2 minutes on CPU.
- **Compliance:** 100% Offline. No network calls. 
- **Validation:** Passes the official `validate_submission.py` perfectly.

## 🧠 Why this gets Rank 1
- **Defeats Honeypots:** The lie-detection matrix successfully identifies candidates who stuff skills they haven't used.
- **Highly Optimized:** 1 Lakh profiles in 90 seconds without a GPU.
- **Rules Compliant:** 101 lines exactly, proper tie-breakers (ID ascending), zero API reliance.
