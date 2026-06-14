import os
import csv
import json
import math
import argparse
import datetime
from typing import List, Dict, Any

import orjson
import numpy as np
from loguru import logger
from pydantic import ValidationError
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from schemas import Candidate
from jd_parser import parse_jd, JDIntentBundle
from enricher import CandidateEnricher
from retrieval import HybridRetriever
from scorer import score_candidate, check_disqualifiers


def sigmoid(x: float) -> float:
    """Standard sigmoid function to map cross-encoder logits to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


class Pipeline:
    def __init__(self, data_path: str, jd_path: str, output_path: str, is_jsonl: bool = True):
        self.data_path = data_path
        self.jd_path = jd_path
        self.output_path = output_path
        self.is_jsonl = is_jsonl

        logger.info("Loading models (sentence-transformers + cross-encoder)...")
        # Using a stronger embedding model for better semantic precision
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.enricher = CandidateEnricher(self.embedding_model)
        self.retriever = HybridRetriever(self.embedding_model)
        self.jd_intent: JDIntentBundle = None
        self.candidates: Dict[str, Candidate] = {}

    # ─────────────────────────────────────────────
    # DATA LOADING
    # ─────────────────────────────────────────────
    def load_data(self):
        # Step 0: Parse JD intent (offline-safe — reads from cache if available)
        self.jd_intent = parse_jd(self.jd_path)
        logger.info(f"JD intent loaded. Semantic anchor length: {len(self.jd_intent.semantic_anchor_text)} chars")

        # Step 1: Load all candidates
        logger.info(f"Loading candidates from: {self.data_path}")
        parse_errors = 0

        if self.is_jsonl:
            with open(self.data_path, "rb") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = orjson.loads(line)
                        cand = Candidate.model_validate(data)
                        self.candidates[cand.candidate_id] = cand
                    except (orjson.JSONDecodeError, ValidationError) as e:
                        parse_errors += 1
                        if parse_errors <= 5:
                            logger.warning(f"Line {line_num}: Parse error — {type(e).__name__}: {e}")
        else:
            with open(self.data_path, "rb") as f:
                data_list = orjson.loads(f.read())
            for data in data_list:
                try:
                    cand = Candidate.model_validate(data)
                    self.candidates[cand.candidate_id] = cand
                except (orjson.JSONDecodeError, ValidationError) as e:
                    parse_errors += 1

        logger.info(f"Loaded {len(self.candidates)} valid candidates. "
                    f"Skipped {parse_errors} malformed records.")

    # ─────────────────────────────────────────────
    # MAIN PIPELINE
    # ─────────────────────────────────────────────
    def run(self):
        self.load_data()

        # ── STAGE 1: Fast BM25 Pre-Filtering (100K → 2000) ────────────────
        logger.info("Stage 1: BM25 Pre-Filtering (100K → top 2000)...")
        cids = list(self.candidates.keys())
        raw_docs = []

        for cid in cids:
            cand = self.candidates[cid]
            text = cand.profile.summary or ""
            text += " " + cand.profile.current_title + " " + cand.profile.current_industry
            if cand.career_history:
                text += " " + " ".join([f"{r.title} {r.description or ''}" for r in cand.career_history])
            if cand.skills:
                text += " " + " ".join([s.name for s in cand.skills])
            raw_docs.append(text.lower().split())

        logger.info("Building BM25 index for pre-filtering...")
        bm25 = BM25Okapi(raw_docs)
        # Use both positive anchor and key skills for BM25 query
        bm25_query = self.jd_intent.semantic_anchor_text.lower().split()
        bm25_scores = bm25.get_scores(bm25_query)

        top_k_prefilter = min(2000, len(cids))
        top_indices = np.argsort(bm25_scores)[::-1][:top_k_prefilter]
        top_2000_cids = [cids[i] for i in top_indices]
        logger.info(f"Pre-filtered to top {len(top_2000_cids)} candidates.")

        # ── STAGE 2: Heavy Enrichment on Top 2000 ─────────────────────────
        logger.info("Stage 2: Heavy enrichment + disqualifier check on top 2000...")
        self.enriched_data: Dict[str, Any] = {}
        docs: List[str] = []
        passed_cids: List[str] = []

        for cid in tqdm(top_2000_cids, desc="Enriching candidates"):
            cand = self.candidates[cid]
            enriched = self.enricher.enrich(cand)
            self.enriched_data[cid] = enriched
            metadata = enriched["metadata_for_scoring"]

            is_hard, _, _ = check_disqualifiers(cand, metadata)
            if not is_hard:
                passed_cids.append(cid)
                docs.append(enriched["vector_document"])

        logger.info(f"Passed disqualifiers: {len(passed_cids)} / {len(top_2000_cids)}")
        if not passed_cids:
            logger.error("No candidates passed Stage 2. Check disqualifier thresholds.")
            return

        # ── STAGE 3: Hybrid Retrieval (Dense FAISS + Sparse BM25) ─────────
        logger.info("Stage 3: Building hybrid index and retrieving top 300...")
        self.retriever.fit(passed_cids, docs)

        retrieved_scores = self.retriever.retrieve(
            query=self.jd_intent.semantic_anchor_text,
            negative_query=self.jd_intent.negative_semantic_anchor,
            top_k=300
        )
        retrieved_cids = list(retrieved_scores.keys())
        logger.info(f"Retrieved {len(retrieved_cids)} candidates for cross-encoder reranking.")

        # ── STAGE 4: Cross-Encoder Reranking ──────────────────────────────
        logger.info("Stage 4: Cross-Encoder reranking on top 300 pairs...")
        pairs = []
        for cid in retrieved_cids:
            cand_doc = self.enriched_data[cid]["vector_document"]
            pairs.append([self.jd_intent.semantic_anchor_text, cand_doc])

        ce_scores_raw = self.cross_encoder.predict(pairs)
        ce_norm = [sigmoid(float(s)) for s in ce_scores_raw]

        final_candidates = []
        for i, cid in enumerate(retrieved_cids):
            hybrid_s = retrieved_scores[cid]       # 0-1 normalized RRF score
            ce_s = ce_norm[i]                       # 0-1 sigmoid of cross-encoder logit

            # Blend: Cross-Encoder is more precise, give it higher weight
            s1_score = (ce_s * 0.65) + (hybrid_s * 0.35)

            cand = self.candidates[cid]
            metadata = self.enriched_data[cid]["metadata_for_scoring"]
            score_results = score_candidate(cand, s1_score, metadata)

            final_candidates.append({
                "candidate_id": cid,
                "score_details": score_results,
                "final_score": score_results["final_score"],
                "candidate": cand,
                "metadata": metadata,
            })

        # Sort by score descending, then candidate_id ascending for deterministic tie-break
        for item in final_candidates:
            item["rounded_score"] = round(item["final_score"], 6)

        final_candidates.sort(key=lambda x: (-x["rounded_score"], x["candidate_id"]))

        # Top 100
        top_100 = final_candidates[:100]

        # ── STAGE 5: Candidate-Specific Reasoning ─────────────────────────
        logger.info("Stage 5: Generating candidate-specific reasoning...")
        top_100 = self.generate_offline_reasoning(top_100)

        # Generate submission CSV
        self.generate_submission(top_100)

    # ─────────────────────────────────────────────
    # CANDIDATE-SPECIFIC REASONING GENERATOR
    # ─────────────────────────────────────────────
    def generate_offline_reasoning(self, top_candidates: List[Dict]) -> List[Dict]:
        """
        Generates honest, candidate-specific reasoning for each ranked entry.
        Avoids generic templates — each string references actual profile data.
        """
        total = len(top_candidates)

        for rank_idx, item in enumerate(top_candidates):
            cand: Candidate = item["candidate"]
            meta: dict = item["metadata"]
            score_det: dict = item["score_details"]
            rank = rank_idx + 1

            # Hard disqualified (should not be here, but defensive)
            if score_det.get("is_hard_dq"):
                reason = " | ".join(score_det.get("dq_reasons", ["Unknown DQ reason"]))
                item["reasoning"] = f"DISQUALIFIED: {reason}"
                continue

            # ── Build honest, specific reasoning ──────────────────────────
            parts = []

            # 1. Years of experience + persona
            yoe = cand.profile.years_of_experience
            persona = meta.get("true_persona", cand.profile.current_title)
            parts.append(f"{yoe:.0f}yr {persona}; currently {cand.profile.current_title} at {cand.profile.current_company}.")

            # 2. Top verified skills (up to 5)
            verified = meta.get("verified_skills", [])[:5]
            if verified:
                parts.append(f"Verified skills: {', '.join(verified)}.")
            hallucinated = meta.get("hallucinated_skills", [])
            if hallucinated:
                parts.append(f"Unverified claims: {', '.join(hallucinated[:3])}.")

            # 3. Semantic fit signal
            s1 = score_det.get("s1_semantic", 0)
            if s1 > 0.75:
                parts.append("Strong semantic match to JD core competencies.")
            elif s1 > 0.55:
                parts.append("Moderate semantic alignment with JD.")
            else:
                parts.append("Weak semantic match; included for other signals.")

            # 4. Company background
            if meta.get("is_pure_consulting"):
                parts.append("Warning: Pure consulting background (no product-company experience).")
            elif meta.get("tier1_product_experience"):
                parts.append("Tier-1 Indian product company experience — strong culture fit signal.")

            # 5. Availability / notice period
            notice = meta.get("notice_period_days", cand.redrob_signals.notice_period_days)
            if notice == 0:
                parts.append("Immediately available.")
            elif notice <= 30:
                parts.append(f"Notice: {notice}d — within buyout range.")
            elif notice <= 60:
                parts.append(f"Notice: {notice}d — acceptable but longer than preferred.")
            else:
                parts.append(f"Notice: {notice}d — concern, above preferred range.")

            # 6. GitHub & external validation
            gh = meta.get("github_activity_score", cand.redrob_signals.github_activity_score)
            if gh == -1:
                parts.append("No GitHub linked — reduced external validation for closed-source work.")
            elif gh >= 70:
                parts.append(f"Strong GitHub activity ({gh}/100) — external proof of work.")
            elif gh >= 30:
                parts.append(f"Moderate GitHub activity ({gh}/100).")

            # 7. Trust score
            trust = meta.get("trust_score", 50)
            if trust >= 80:
                parts.append(f"High trust ({trust:.0f}%) — skills well-supported by career history.")
            elif trust < 30:
                parts.append(f"Low trust ({trust:.0f}%) — many claimed skills not verifiable in work history.")

            # 8. Behavioral / engagement
            vibe = meta.get("behavioral_score", 60)
            if vibe >= 85:
                parts.append("High platform engagement — responsive and actively seeking.")
            elif vibe < 50:
                parts.append("Low platform engagement — may be hard to reach.")

            # 9. Soft DQ warnings
            dq_reasons = score_det.get("dq_reasons", [])
            soft_dqs = [r for r in dq_reasons if not r.startswith("Hard")]
            if soft_dqs:
                parts.append(f"Concerns: {'; '.join(soft_dqs[:2])}.")

            # 10. Rank context
            if rank <= 10:
                parts.append("Top-10 candidate — strong overall fit.")
            elif rank <= 25:
                parts.append("Strong candidate overall.")
            elif rank > 75:
                parts.append("Adjacent skills — included near cutoff for broad coverage.")

            item["reasoning"] = " ".join(parts)

        return top_candidates

    # ─────────────────────────────────────────────
    # SUBMISSION CSV WRITER
    # ─────────────────────────────────────────────
    def generate_submission(self, top_100: List[Dict]):
        logger.info(f"Writing final submission CSV to: {self.output_path}")

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])

            for rank, item in enumerate(top_100, 1):
                cid = item["candidate_id"]
                score = round(item["final_score"], 4)
                reasoning = item.get("reasoning", "")
                writer.writerow([cid, rank, f"{score:.4f}", reasoning])

        logger.info(f"Done! {len(top_100)} candidates written to {self.output_path}")

        # Print score summary for debugging
        scores = [round(item["final_score"], 4) for item in top_100]
        logger.info(f"Score range: [{min(scores):.4f} — {max(scores):.4f}]")
        logger.info(f"Top 5 scores: {scores[:5]}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Recruiter Pipeline")
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl",
                        help="Path to candidates data file")
    parser.add_argument("--jd", type=str, default="data/jd_extracted.txt",
                        help="Path to JD text file")
    parser.add_argument("--output", type=str, default="solo_participant.csv",
                        help="Output CSV path")
    parser.add_argument("--format", type=str, choices=["json", "jsonl"], default="jsonl",
                        help="Format of the candidates file")

    args = parser.parse_args()
    is_jsonl = (args.format == "jsonl" or args.candidates.endswith(".jsonl"))

    pipeline = Pipeline(args.candidates, args.jd, args.output, is_jsonl=is_jsonl)
    pipeline.run()
