# Smart Recruiter: AI-Powered Applicant Tracking System

## 🚀 Overview

**Smart Recruiter** is an advanced, AI-driven Applicant Tracking System (ATS) pipeline designed to bridge the gap between complex job requirements and candidate profiles. By moving beyond traditional keyword matching, this system deeply understands the *intent* of a job description and evaluates candidate fit based on semantic meaning, verified skills, and explicit disqualifiers.

## 🛑 The Problem with Traditional Keyword-Based ATS

Traditional ATS platforms rely heavily on simplistic keyword matching, which leads to several major issues:
1. **Missed Context:** They fail to differentiate between "managed a team of software engineers" and "managed by a team of software engineers."
2. **Keyword Stuffing:** Candidates who stuff their resumes with buzzwords often rank higher than genuinely qualified candidates who use different terminology.
3. **Inability to Infer:** If a JD asks for "Python," a candidate listing "Pandas, FastAPI, and PyTorch" might be rejected if they didn't explicitly write the word "Python."
4. **Lack of Human-like Reasoning:** They cannot evaluate nuanced criteria like "must have startup experience" or "requires a background in scaling distributed systems."

## 💡 How Smart Recruiter Solves the Problem

Smart Recruiter utilizes a multi-stage AI pipeline to emulate the reasoning of a Principal Technical Recruiter:
- **Deep Intent Parsing:** Extracts core semantic anchors, hard constraints, and soft preferences directly from the Job Description.
- **Semantic Understanding:** Uses dense vector embeddings to match the meaning behind a candidate's experience with the JD, defeating keyword stuffing and inferring implicit skills.
- **Multi-Stage Funnel:** Employs a fast Hybrid Retriever for initial screening, a precise Cross-Encoder for deep semantic reranking, and a Large Language Model (LLM) Judge for final qualitative assessment.
- **Transparent Reasoning:** Instead of just outputting a black-box score, the LLM Judge provides a professional, human-readable explanation of *why* a candidate is a good fit or why they were disqualified.

## 🏗️ Architecture & Pipeline

The pipeline is composed of four main stages:

1. **Stage 1: Candidate Enrichment & Hard Filtering (`enricher.py` / `jd_parser.py`)**
   - Parses the JD into structured intents (skills, constraints, disqualifiers).
   - Enriches candidate JSON profiles into dense text documents.
   - Applies hard filtering (e.g., automatically disqualifying candidates who violate explicit geographic or visa constraints).

2. **Stage 2: Hybrid Retrieval (`retrieval.py`)**
   - Combines Dense Retrieval (Sentence Transformers, e.g., `all-MiniLM-L6-v2`) and Sparse Retrieval (BM25) to quickly narrow down a large pool of candidates to the top 500 semantic matches.

3. **Stage 3: Cross-Encoder Rerank (`scorer.py`)**
   - Runs the top candidates through a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) which evaluates the JD intent and candidate profile simultaneously, producing a highly accurate semantic fit score (S1 Score).

4. **Stage 4: LLM Judge & Reasoning (`pipeline.py`)**
   - The top 100 candidates are sent to an LLM (Llama-3.3-70b via Groq API).
   - The LLM acts as a recruiter, checking for nuanced disqualifiers and outputting a final weighted score along with a 2-sentence professional pitch justifying the decision.

## 📊 Dataset Information

The system operates on structural data. It requires:
1. **Candidates Data (`candidate.jsonl`):** 🌟 **HIGHLIGHT: This dataset contains 1 Lakh (100,000) real candidate/student profiles.** It includes deep structural fields such as ID, name, work experience, skills, and education. Processing real-world profiles at this scale ensures the pipeline is robust and highly scalable.
2. **Job Description (Text):** A raw text file containing the comprehensive job requirements and intent.

## ⚙️ Installation & Usage

### Prerequisites
- Python 3.9+
- A [Groq](https://groq.com/) API Key for the LLM Judge

### Setup
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Setup environment variables:
   Create a `.env` file in the root directory and add your API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

### Running the Pipeline
Run the pipeline via the CLI by specifying the candidate dataset and the JD text file:

```bash
python pipeline.py --candidates data/sample_candidates.json --jd data/jd_extracted.txt --output submission.csv --format json
```

The output will be generated as a CSV file (`submission.csv`) containing the candidate ID, rank, final score, and the LLM's reasoning for each candidate.
