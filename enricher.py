import re
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import faiss

# --- HELPER CONSTANTS ---

class CandidateEnricher:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
        # Reference personas for dynamic extraction
        self.personas = [
            "Data Engineer",
            "Applied ML Engineer",
            "Backend Engineer",
            "Frontend Engineer",
            "Product Manager",
            "Software Engineer",
            "Research Scientist"
        ]
        self.persona_embeddings = self.embedding_model.encode(self.personas, convert_to_numpy=True)
        faiss.normalize_L2(self.persona_embeddings)

    def _extract_true_persona(self, career_history: List[Any]) -> str:
        """Derives actual role using Semantic Similarity to standard personas."""
        combined_desc = " ".join([role.description.lower() for role in career_history if role.description])
        if not combined_desc.strip():
            return "Software Engineer" # Fallback
            
        desc_emb = self.embedding_model.encode([combined_desc], convert_to_numpy=True)
        faiss.normalize_L2(desc_emb)
        
        similarities = np.dot(desc_emb, self.persona_embeddings.T)[0]
        best_idx = np.argmax(similarities)
        return self.personas[best_idx]

    def _verify_skills(self, skills: List[Any], combined_text: str) -> Dict[str, List[str]]:
        """The Smart Lie Detector: Checks if claimed skills are semantically proven in work history."""
        verified = []
        hallucinated = []
        
        if not combined_text.strip() or not skills:
            return {"verified": [], "hallucinated": [s.name for s in skills]}
            
        # Split text into sentences for fine-grained semantic matching
        sentences = [s.strip() for s in combined_text.replace('\n', '.').split('.') if len(s.strip()) > 10]
        if not sentences:
            sentences = [combined_text]
            
        sent_embs = self.embedding_model.encode(sentences, convert_to_numpy=True)
        faiss.normalize_L2(sent_embs)
        
        skill_names = [s.name for s in skills]
        skill_embs = self.embedding_model.encode(skill_names, convert_to_numpy=True)
        faiss.normalize_L2(skill_embs)
        
        # similarity matrix: (num_skills, num_sentences)
        similarities = np.dot(skill_embs, sent_embs.T)
        max_sims = np.max(similarities, axis=1) # max similarity across all sentences
        
        for i, skill in enumerate(skills):
            skill_lower = skill.name.lower()
            text_lower = combined_text.lower()
            
            # Direct mention OR high semantic similarity with at least one sentence (threshold 0.45)
            if skill_lower in text_lower or max_sims[i] > 0.45:
                verified.append(skill.name)
            else:
                hallucinated.append(skill.name)
                
        return {"verified": verified, "hallucinated": hallucinated}

    def _calculate_trust_score(self, verified_count: int, total_count: int) -> float:
        if total_count == 0: return 0.0
        return round((verified_count / total_count) * 100, 2)

    def _evaluate_logistics_and_dealbreakers(self, candidate) -> Dict[str, Any]:
        """Evaluates Hard JD constraints."""
        profile = candidate.profile
        signals = candidate.redrob_signals
        
        flags = {
            "is_dealbreaker": False,
            "reasons": []
        }
        
        # 1. Location & Relocation Check
        # Removed hardcoded country constraints to prevent unfair disqualification of remote/international candidates.
        # Downstream systems or LLM will evaluate location constraints.
            
        # Removed hardcoded Company DNA Check (Service vs Product). Let LLM Judge handle this dynamically.
            
        # 3. Notice Period Check
        if signals.notice_period_days > 45:
            flags["reasons"].append(f"High Notice Period: {signals.notice_period_days} days.")
            
        return flags

    def _evaluate_behavioral_vibe(self, signals) -> float:
        """Scores candidate based on platform activity."""
        score = 100.0
        
        # Ghosting Penalty
        if signals.recruiter_response_rate < 0.50:
            score -= 30
        
        # Slow Responder Penalty (e.g. > 48 hours / 2 days)
        if signals.avg_response_time_hours > 48:
            score -= 20
            
        # Open to work bonus
        if signals.open_to_work_flag:
            score += 10
            
        return max(0.0, min(100.0, score)) # Clamp between 0 and 100

    def build_truthful_document(self, candidate, verified_skills: List[str], true_persona: str) -> str:
        """Builds a semantic document using ONLY verified truths for vector search."""
        doc_parts = []
        
        # Inject the derived truth, not the claimed title
        doc_parts.append(f"True Engineering Persona: {true_persona}")
        doc_parts.append(f"Reported Summary: {candidate.profile.summary}")
        
        doc_parts.append("Verified Production Experience:")
        for role in candidate.career_history:
            doc_parts.append(f"- At {role.company} ({role.industry}): {role.description}")
            
        # ONLY inject verified skills into the embedding space
        if verified_skills:
            doc_parts.append(f"Verified Applied Skills: {', '.join(verified_skills)}")
        else:
            doc_parts.append("Verified Applied Skills: None explicitly found in work history.")
            
        return " \n".join(doc_parts)

    def enrich(self, candidate) -> Dict[str, Any]:
        """
        Master enrichment function. Returns a multi-dimensional dictionary.
        """
        # Combine all work text for lie detection
        work_text = " ".join([role.description for role in candidate.career_history if role.description])
        work_text += " " + candidate.profile.summary
        
        # 1. The Lie Detector (Skill Verification)
        skill_verification = self._verify_skills(candidate.skills, work_text)
        trust_score = self._calculate_trust_score(len(skill_verification["verified"]), len(candidate.skills))
        
        # 2. Derive True Persona
        true_persona = self._extract_true_persona(candidate.career_history)
        
        # 3. Logistics & Dealbreakers
        logistics_flags = self._evaluate_logistics_and_dealbreakers(candidate)
        
        # 4. Behavioral Vibe
        behavioral_score = self._evaluate_behavioral_vibe(candidate.redrob_signals)
        
        # 5. Build the Clean Document (No hallucinations)
        truthful_document = self.build_truthful_document(candidate, skill_verification["verified"], true_persona)
        
        # The Final Enriched Object
        return {
            "candidate_id": candidate.candidate_id,
            "vector_document": truthful_document, # Give this to BM25 / Embedding Model
            "metadata_for_scoring": {
                "trust_score": trust_score,           # 0 to 100
                "behavioral_score": behavioral_score, # 0 to 100
                "true_persona": true_persona,
                "hallucinated_skills_count": len(skill_verification["hallucinated"]),
                "is_dealbreaker": logistics_flags["is_dealbreaker"],
                "dealbreaker_reasons": logistics_flags["reasons"]
            }
        }