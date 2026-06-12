import os
import csv
import json
import orjson
import argparse
from typing import List, Dict, Any
from pydantic import ValidationError
from loguru import logger
from tqdm import tqdm
from groq import Groq
from dotenv import load_dotenv

from schemas import Candidate
from jd_parser import parse_jd, JDIntentBundle
from enricher import CandidateEnricher
from retrieval import HybridRetriever
from scorer import score_candidate, check_disqualifiers
from sentence_transformers import CrossEncoder

load_dotenv()

class Pipeline:
    def __init__(self, data_path: str, jd_path: str, output_path: str, is_jsonl: bool = True):
        self.data_path = data_path
        self.jd_path = jd_path
        self.output_path = output_path
        self.is_jsonl = is_jsonl
        
        logger.info("Loading Models...")
        from sentence_transformers import SentenceTransformer
        # Switched to a smaller model for the trial run since 1.3GB was hanging on download
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.enricher = CandidateEnricher(self.embedding_model)
        self.retriever = HybridRetriever(self.embedding_model)
        self.jd_intent: JDIntentBundle = None
        self.candidates: Dict[str, Candidate] = {}
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            logger.error("CRITICAL: GROQ_API_KEY not found in environment variables.")
            raise ValueError("GROQ_API_KEY is required for the LLM Judge. Please add it to your .env file.")
        self.groq_client = Groq(api_key=api_key)

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
                        logger.warning(f"Skipping row due to error: {e}")
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

    # Removed stage1_hard_filter method as we will do it inline after enrichment

    def run(self):
        self.load_data()
        
        self.enriched_data = {}
        docs = []
        passed_cids = []
        
        logger.info("Running Stage 1 & 2: Enrichment, Hard Filtering, and Indexing")
        for cid, cand in self.candidates.items():
            enriched = self.enricher.enrich(cand)
            self.enriched_data[cid] = enriched
            metadata = enriched["metadata_for_scoring"]
            
            # Apply Hard Disqualifiers using new metadata
            is_hard, _, _ = check_disqualifiers(cand, metadata)
            if not is_hard:
                passed_cids.append(cid)
                docs.append(enriched["vector_document"])
                
        logger.info(f"Stage 1 passed: {len(passed_cids)} / {len(self.candidates)}")
        
        if not passed_cids:
            logger.error("No candidates passed Stage 1.")
            return
            
        self.retriever.fit(passed_cids, docs)
        
        # Query using positive and negative semantic anchors
        query = self.jd_intent.semantic_anchor_text
        neg_query = getattr(self.jd_intent, "negative_semantic_anchor", None)
        
        retrieved_scores = self.retriever.retrieve(query, negative_query=neg_query, top_k=500)
        
        retrieved_cids = list(retrieved_scores.keys())
        logger.info(f"Stage 2 retrieved top {len(retrieved_cids)} candidates.")

        # Stage 3: Cross-Encoder Rerank
        logger.info("Running Stage 3: Cross-Encoder Rerank")
        pairs = []
        for cid in retrieved_cids:
            cand_doc = self.enriched_data[cid]["vector_document"]
            pairs.append([query, cand_doc])
            
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Combine Retrieval + CE scores into Semantic Fit (S1)
        import math
        def sigmoid(x):
            # Standard sigmoid for more even score distribution across CE logits
            return 1 / (1 + math.exp(-x))
            
        ce_norm = [sigmoid(s) for s in ce_scores]
            
        final_candidates = []
        for i, cid in enumerate(retrieved_cids):
            hybrid_s = retrieved_scores[cid]
            ce_s = ce_norm[i]
            
            # S1 computation (approximate, ignoring LLM judge for speed, or LLM judge can be mocked)
            # S1 = LLM * 0.5 + CE * 0.3 + Hybrid * 0.2
            # S1 computation: S1 = CE * 0.6 + Hybrid * 0.4 (Pre-LLM)
            s1_score = (ce_s * 0.6) + (hybrid_s * 0.4)
            
            cand = self.candidates[cid]
            metadata = self.enriched_data[cid]["metadata_for_scoring"]
            score_results = score_candidate(cand, s1_score, metadata)
            
            final_candidates.append({
                "candidate_id": cid,
                "score_details": score_results,
                "final_score": score_results["final_score"],
                "candidate": cand,
                "vector_document": self.enriched_data[cid]["vector_document"]
            })
            
        # Sort by final score
        final_candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        
        # Take Top 100 for LLM Judge
        top_100 = final_candidates[:100]
        
        # Stage 4: LLM Judge & Reasoning Generator for Top 100
        top_100 = self.run_llm_judge(top_100)
        
        # Re-sort after LLM Judge
        top_100.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
        
        # Generate Output
        self.generate_submission(top_100)

    def run_llm_judge(self, top_candidates: List[Dict]) -> List[Dict]:
        logger.info("Running LLM Judge on Top Candidates to finalize scores and generate reasoning...")
        intent_json = self.jd_intent.model_dump_json(indent=2)
        
        for item in tqdm(top_candidates, desc="Evaluating with LLM"):
            cand = item["candidate"]
            doc = item["vector_document"]
            
            prompt = f"""
            You are a Principal Technical Recruiter. Evaluate this candidate against the comprehensive job intent rules.
            
            JOB INTENT RULES:
            {intent_json}
            
            Candidate True Persona & Verified Background:
            {doc}
            
            Provide a JSON output ONLY with exactly these keys:
            "is_disqualified": Boolean. True if they explicitly violate "temporal_constraints" or "skill_dependencies" or any "banned_backgrounds".
            "disqualification_reason": String. If disqualified, state exactly which rule they failed. Else empty string.
            "llm_fit_score": A float between 0.0 and 1.0 representing how perfectly they match the core competencies and environment context.
            "recruiter_reasoning": A crisp, professional 2-sentence pitch explaining exactly WHY this candidate is a strong fit (or why they failed) based on their verified background mapping to the specific JD Intent rules. Do not use generic fluff.
            """
            
            try:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                res_data = json.loads(response.choices[0].message.content)
                is_dq = res_data.get("is_disqualified", False)
                llm_score = float(res_data.get("llm_fit_score", 0.8))
                reasoning = str(res_data.get("recruiter_reasoning", "Candidate is a strong semantic fit."))
                
                if is_dq:
                    item["final_score"] = 0.0
                    item["reasoning"] = f"DISQUALIFIED: {res_data.get('disqualification_reason', 'Failed JD rules.')} | {reasoning}"
                else:
                    item["final_score"] = (item["final_score"] * 0.7) + (llm_score * 0.3)
                    item["reasoning"] = reasoning
                
            except Exception as e:
                logger.debug(f"LLM Judge failed for {cand.candidate_id}: {e}")
                item["reasoning"] = "Candidate passed all algorithmic stages."
                
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
    parser.add_argument("--candidates", type=str, default="data/sample_candidates.json", help="Path to candidates data")
    parser.add_argument("--jd", type=str, default="data/jd_extracted.txt", help="Path to JD text file")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV path")
    parser.add_argument("--format", type=str, choices=["json", "jsonl"], default="json", help="Format of the candidates file")
    
    args = parser.parse_args()
    
    is_jsonl = (args.format == "jsonl" or args.candidates.endswith(".jsonl"))
    
    pipeline = Pipeline(args.candidates, args.jd, args.output, is_jsonl=is_jsonl)
    pipeline.run()
