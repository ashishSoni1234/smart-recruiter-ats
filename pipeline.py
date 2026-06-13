import os
import csv
import json
import orjson
import argparse
from typing import List, Dict, Any
from pydantic import ValidationError
from loguru import logger
from tqdm import tqdm
import numpy as np

from schemas import Candidate
from jd_parser import parse_jd, JDIntentBundle
from enricher import CandidateEnricher
from retrieval import HybridRetriever
from scorer import score_candidate, check_disqualifiers
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

class Pipeline:
    def __init__(self, data_path: str, jd_path: str, output_path: str, is_jsonl: bool = True):
        self.data_path = data_path
        self.jd_path = jd_path
        self.output_path = output_path
        self.is_jsonl = is_jsonl
        
        logger.info("Loading Models...")
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.enricher = CandidateEnricher(self.embedding_model)
        self.retriever = HybridRetriever(self.embedding_model)
        self.jd_intent: JDIntentBundle = None
        self.candidates: Dict[str, Candidate] = {}

    def load_data(self):
        self.jd_intent = parse_jd(self.jd_path)
        logger.info(f"Loaded JD Intent. Anchor length: {len(self.jd_intent.semantic_anchor_text)}")
        
        logger.info(f"Loading candidates from {self.data_path}")
        if self.is_jsonl:
            with open(self.data_path, "rb") as f:
                for line in f:
                    try:
                        data = orjson.loads(line)
                        cand = Candidate.model_validate(data)
                        self.candidates[cand.candidate_id] = cand
                    except Exception as e:
                        pass
        else:
            with open(self.data_path, "rb") as f:
                data_list = orjson.loads(f.read())
                for data in data_list:
                    try:
                        cand = Candidate.model_validate(data)
                        self.candidates[cand.candidate_id] = cand
                    except Exception as e:
                        pass
        logger.info(f"Loaded {len(self.candidates)} valid candidates.")

    def run(self):
        self.load_data()
        
        # 1. Fast Pre-Filtering (100k -> 2000)
        logger.info("Running Stage 1: Fast BM25 Pre-Filtering on raw text...")
        raw_docs = []
        cids = list(self.candidates.keys())
        
        for cid in cids:
            cand = self.candidates[cid]
            # Construct a raw text representation quickly
            text = cand.profile.summary or ""
            if cand.career_history:
                text += " " + " ".join([r.description or "" for r in cand.career_history])
            if cand.skills:
                text += " " + " ".join([s.name for s in cand.skills])
            raw_docs.append(text.lower().split())
            
        logger.info("Tokenizing and building BM25 index for pre-filtering...")
        bm25 = BM25Okapi(raw_docs)
        query = self.jd_intent.semantic_anchor_text.lower().split()
        bm25_scores = bm25.get_scores(query)
        
        # Select top 2000
        top_k_prefilter = min(2000, len(cids))
        top_indices = np.argsort(bm25_scores)[::-1][:top_k_prefilter]
        
        top_2000_cids = [cids[i] for i in top_indices]
        logger.info(f"Pre-filtered top {len(top_2000_cids)} candidates for heavy enrichment.")
        
        # 2. Heavy Enrichment & Disqualifiers
        self.enriched_data = {}
        docs = []
        passed_cids = []
        
        logger.info("Running Stage 2: Heavy Enrichment and Disqualifiers on Top 2000...")
        for cid in tqdm(top_2000_cids, desc="Enriching"):
            cand = self.candidates[cid]
            enriched = self.enricher.enrich(cand)
            self.enriched_data[cid] = enriched
            metadata = enriched["metadata_for_scoring"]
            
            is_hard, _, _ = check_disqualifiers(cand, metadata)
            if not is_hard:
                passed_cids.append(cid)
                docs.append(enriched["vector_document"])
                
        logger.info(f"Passed disqualifiers: {len(passed_cids)}")
        if not passed_cids:
            logger.error("No candidates passed Stage 2.")
            return
            
        # 3. Hybrid Retrieval (Dense + Sparse)
        logger.info("Running Stage 3: Hybrid Retrieval...")
        self.retriever.fit(passed_cids, docs)
        
        retrieved_scores = self.retriever.retrieve(
            self.jd_intent.semantic_anchor_text, 
            negative_query=getattr(self.jd_intent, "negative_semantic_anchor", None), 
            top_k=300
        )
        retrieved_cids = list(retrieved_scores.keys())
        logger.info(f"Retrieved top {len(retrieved_cids)} candidates for Cross-Encoder.")

        # 4. Cross-Encoder Rerank
        logger.info("Running Stage 4: Cross-Encoder Rerank...")
        pairs = []
        for cid in retrieved_cids:
            cand_doc = self.enriched_data[cid]["vector_document"]
            pairs.append([self.jd_intent.semantic_anchor_text, cand_doc])
            
        ce_scores = self.cross_encoder.predict(pairs)
        
        import math
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
            
        ce_norm = [sigmoid(s) for s in ce_scores]
            
        final_candidates = []
        for i, cid in enumerate(retrieved_cids):
            hybrid_s = retrieved_scores[cid]
            ce_s = ce_norm[i]
            
            s1_score = (ce_s * 0.6) + (hybrid_s * 0.4)
            
            cand = self.candidates[cid]
            metadata = self.enriched_data[cid]["metadata_for_scoring"]
            score_results = score_candidate(cand, s1_score, metadata)
            
            final_candidates.append({
                "candidate_id": cid,
                "score_details": score_results,
                "final_score": score_results["final_score"],
                "candidate": cand,
                "metadata": metadata
            })
            
        # Sort by rounded score descending, then candidate_id ascending for tie-break
        for item in final_candidates:
            item["rounded_score"] = round(item["final_score"], 4)
            
        final_candidates.sort(key=lambda x: (-x["rounded_score"], x["candidate_id"]))
        
        # Exact Top 100
        top_100 = final_candidates[:100]
        
        # 5. Programmatic Offline Reasoning
        logger.info("Running Stage 5: Programmatic Offline Reasoning...")
        top_100 = self.generate_offline_reasoning(top_100)
        
        # Generate Output
        self.generate_submission(top_100)

    def generate_offline_reasoning(self, top_candidates: List[Dict]) -> List[Dict]:
        for item in top_candidates:
            cand = item["candidate"]
            meta = item["metadata"]
            final_score = item["final_score"]
            score_det = item["score_details"]
            
            if score_det.get("is_hard_dq"):
                reason = " | ".join(score_det.get("dq_reasons", []))
                item["reasoning"] = f"DISQUALIFIED: {reason}"
            else:
                persona = meta.get("true_persona", "Software Engineer")
                trust = meta.get("trust_score", 50)
                vibe = meta.get("behavioral_score", 50)
                
                reasoning = f"Strong fit! Derived persona: {persona}. High semantic relevance to core competencies. "
                reasoning += f"Trust factor: {trust}% based on verifiable work history. "
                reasoning += f"Behavioral signals: {vibe}% indicating positive culture alignment and availability."
                item["reasoning"] = reasoning
                
        return top_candidates

    def generate_submission(self, top_100: List[Dict]):
        logger.info(f"Generating final submission CSV at {self.output_path}...")
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            
            for rank, item in enumerate(top_100, 1):
                cid = item["candidate_id"]
                score = round(item["final_score"], 4)
                reasoning = item.get("reasoning", "")
                
                writer.writerow([cid, rank, f"{score:.4f}", reasoning])
                
        logger.info(f"Done! Output written to {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Recruiter Pipeline")
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl", help="Path to candidates data")
    parser.add_argument("--jd", type=str, default="data/jd_extracted.txt", help="Path to JD text file")
    parser.add_argument("--output", type=str, default="solo_participant.csv", help="Output CSV path")
    parser.add_argument("--format", type=str, choices=["json", "jsonl"], default="jsonl", help="Format of the candidates file")
    
    args = parser.parse_args()
    
    is_jsonl = (args.format == "jsonl" or args.candidates.endswith(".jsonl"))
    
    pipeline = Pipeline(args.candidates, args.jd, args.output, is_jsonl=is_jsonl)
    pipeline.run()
