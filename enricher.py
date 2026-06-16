import re
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import faiss

# ─────────────────────────────────────────────
# CONSULTING / SERVICES COMPANY DETECTOR
# ─────────────────────────────────────────────
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "l&t infotech",
    "mindtree", "persistent", "niit technologies", "kpit", "cyient"
}

# Tier-1 Indian product companies (bonus for product experience)
PRODUCT_COMPANIES_TIER1 = {
    "flipkart", "swiggy", "zomato", "ola", "paytm", "meesho", "zepto",
    "razorpay", "freshworks", "browserstack", "chargebee", "clevertap",
    "urban company", "cred", "groww", "zerodha", "navi", "sharechat",
    "dailyhunt", "vedantu", "unacademy", "byju"
}


class CandidateEnricher:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

        # Extended persona list — now includes AI/ML roles explicitly
        self.personas = [
            "Applied ML Engineer",
            "AI Engineer",
            "Machine Learning Engineer",
            "Data Scientist",
            "Data Engineer",
            "Backend Software Engineer",
            "Frontend Engineer",
            "Product Manager",
            "Research Scientist",
            "DevOps Engineer",
            "Software Engineer",
        ]
        self.persona_embeddings = self.embedding_model.encode(self.personas, convert_to_numpy=True)
        faiss.normalize_L2(self.persona_embeddings)

    # ─────────────────────────────────────────────
    # 1. TRUE PERSONA EXTRACTION
    # ─────────────────────────────────────────────
    def _extract_true_persona(self, career_history: List[Any]) -> str:
        """Derives actual role using Semantic Similarity to standard personas."""
        # Combine titles AND descriptions for richer signal
        combined_desc = " ".join(
            [f"{role.title} {role.description}".lower()
             for role in career_history if role.description]
        )
        if not combined_desc.strip():
            return "Software Engineer"

        desc_emb = self.embedding_model.encode([combined_desc], convert_to_numpy=True)
        faiss.normalize_L2(desc_emb)

        similarities = np.dot(desc_emb, self.persona_embeddings.T)[0]
        best_idx = int(np.argmax(similarities))
        return self.personas[best_idx]

    # ─────────────────────────────────────────────
    # 2. SKILL VERIFICATION (LIE DETECTOR)
    # ─────────────────────────────────────────────
    def _verify_skills(self, skills: List[Any], combined_text: str) -> Dict[str, List[str]]:
        """
        Checks if claimed skills are semantically proven in work history.
        Uses a LOWER threshold (0.30) to reduce false hallucinations.
        Direct keyword match always wins regardless of embedding similarity.
        """
        if not combined_text.strip() or not skills:
            return {"verified": [], "hallucinated": [s.name for s in skills]}

        text_lower = combined_text.lower()

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
        max_sims = np.max(similarities, axis=1)

        verified = []
        hallucinated = []

        for i, skill in enumerate(skills):
            skill_lower = skill.name.lower()

            # Direct keyword match (very reliable) OR semantic similarity >= 0.30
            if skill_lower in text_lower or max_sims[i] > 0.30:
                verified.append(skill.name)
            else:
                # Check partial word match for compound skills (e.g. "vector DB" in text)
                words = skill_lower.split()
                if any(w in text_lower for w in words if len(w) > 3):
                    verified.append(skill.name)
                else:
                    hallucinated.append(skill.name)

        return {"verified": verified, "hallucinated": hallucinated}

    # ─────────────────────────────────────────────
    # 3. TRUST SCORE
    # ─────────────────────────────────────────────
    def _calculate_trust_score(self, verified_count: int, total_count: int) -> float:
        if total_count == 0:
            return 50.0  # Neutral when no skills listed
        return round((verified_count / total_count) * 100, 2)

    # ─────────────────────────────────────────────
    # 4. LOGISTICS & HARD DEALBREAKERS
    # ─────────────────────────────────────────────
    def _evaluate_logistics_and_dealbreakers(self, candidate) -> Dict[str, Any]:
        """Evaluates hard JD constraints."""
        profile = candidate.profile
        signals = candidate.redrob_signals

        flags = {
            "is_dealbreaker": False,
            "reasons": []
        }

        # Hard DQ: Notice period > 90 days (JD says they can buy out up to 30 days,
        # 30+ is still in scope but bar is higher — only hard-DQ above 90)
        if signals.notice_period_days > 90:
            flags["is_dealbreaker"] = True
            flags["reasons"].append(f"Hard DQ: Notice period {signals.notice_period_days} days (>90 days).")

        # Soft Warning: 31–90 days notice (will be handled as score penalty, not DQ)
        elif signals.notice_period_days > 30:
            flags["reasons"].append(f"Soft Warning: Notice period {signals.notice_period_days} days (>30 days preferred).")

        # Honeypot detection: impossible experience
        if hasattr(profile, 'years_of_experience'):
            total_career_months = sum(r.duration_months for r in candidate.career_history)
            if total_career_months > 0:
                claimed_months = int(profile.years_of_experience * 12)
                # If claimed experience is >24 months more than sum of all roles — likely honeypot
                if claimed_months > total_career_months + 24:
                    flags["is_dealbreaker"] = True
                    flags["reasons"].append(
                        f"Honeypot Detected: Claims {profile.years_of_experience:.1f} yrs "
                        f"but career history only spans {total_career_months/12:.1f} yrs."
                    )

        # Honeypot detection: "expert" in >=6 skills with 0 endorsements and 0 duration
        expert_skills = [s for s in candidate.skills if s.proficiency == "expert"]
        if len(expert_skills) >= 6:
            zero_proof = [s for s in expert_skills if s.endorsements == 0 and (s.duration_months or 0) == 0]
            if len(zero_proof) >= 4:
                flags["is_dealbreaker"] = True
                flags["reasons"].append(
                    f"Honeypot Detected: Claims {len(expert_skills)} 'expert' skills "
                    f"with {len(zero_proof)} having zero endorsements and zero duration."
                )

        return flags

    # ─────────────────────────────────────────────
    # 5. BEHAVIORAL VIBE SCORE
    # ─────────────────────────────────────────────
    def _evaluate_behavioral_vibe(self, signals) -> float:
        """Comprehensive platform engagement score using all available signals."""
        score = 60.0  # Start at neutral, not 100

        # ── Responsiveness ──────────────────────
        if signals.recruiter_response_rate >= 0.8:
            score += 15
        elif signals.recruiter_response_rate >= 0.5:
            score += 5
        elif signals.recruiter_response_rate < 0.3:
            score -= 20
        elif signals.recruiter_response_rate < 0.5:
            score -= 10

        # ── Response speed ───────────────────────
        if signals.avg_response_time_hours <= 12:
            score += 5
        elif signals.avg_response_time_hours > 48:
            score -= 10

        # ── Active job seeker signals ─────────────
        if signals.open_to_work_flag:
            score += 10

        if signals.applications_submitted_30d >= 3:
            score += 5

        # ── Platform engagement ───────────────────
        if signals.profile_views_received_30d >= 10:
            score += 5

        if signals.saved_by_recruiters_30d >= 3:
            score += 8  # Social proof — recruiters find value

        if signals.search_appearance_30d >= 20:
            score += 3

        # ── Interview integrity ───────────────────
        if signals.interview_completion_rate >= 0.8:
            score += 8
        elif signals.interview_completion_rate < 0.5:
            score -= 10

        # ── Profile quality ────────────────────────
        if signals.profile_completeness_score >= 80:
            score += 5
        elif signals.profile_completeness_score < 50:
            score -= 5

        # ── Identity verification ─────────────────
        if signals.verified_email and signals.verified_phone:
            score += 5
        if signals.linkedin_connected:
            score += 3

        # ── Offer acceptance (commitment signal) ──
        if signals.offer_acceptance_rate > 0.7:
            score += 5
        elif signals.offer_acceptance_rate >= 0 and signals.offer_acceptance_rate < 0.3:
            score -= 5

        return max(0.0, min(100.0, score))

    # ─────────────────────────────────────────────
    # 6. GITHUB & PLATFORM SKILL SCORES
    # ─────────────────────────────────────────────
    def _evaluate_external_validation(self, signals) -> float:
        """
        Scores external proof of work — critical per JD for closed-source workers.
        Returns a 0-100 score.
        """
        score = 30.0  # Neutral base

        # GitHub activity (JD explicitly calls this out as a proxy proof)
        gh = signals.github_activity_score
        if gh == -1:
            score -= 5   # No GitHub linked — minor penalty
        elif gh >= 70:
            score += 40
        elif gh >= 40:
            score += 25
        elif gh >= 10:
            score += 10

        # Platform skill assessment scores (objective, platform-validated)
        if signals.skill_assessment_scores:
            avg_assessment = sum(signals.skill_assessment_scores.values()) / len(signals.skill_assessment_scores)
            if avg_assessment >= 80:
                score += 30
            elif avg_assessment >= 60:
                score += 20
            elif avg_assessment >= 40:
                score += 10

        # Community signals
        if signals.endorsements_received >= 50:
            score += 10
        elif signals.endorsements_received >= 20:
            score += 5

        if signals.connection_count >= 200:
            score += 5

        return max(0.0, min(100.0, score))

    # ─────────────────────────────────────────────
    # 7. PRODUCT vs CONSULTING BACKGROUND
    # ─────────────────────────────────────────────
    def _evaluate_company_dna(self, candidate) -> Dict[str, Any]:
        """Checks if candidate comes from product vs pure consulting background."""
        consulting_count = 0
        product_count = 0
        tier1_product = False

        for role in candidate.career_history:
            company_lower = role.company.lower()
            industry_lower = role.industry.lower()

            if any(firm in company_lower for firm in CONSULTING_FIRMS):
                consulting_count += 1
            elif any(p in company_lower for p in PRODUCT_COMPANIES_TIER1):
                product_count += 1
                tier1_product = True
            elif "product" in industry_lower or "saas" in industry_lower or "startup" in industry_lower:
                product_count += 1

        total_roles = len(candidate.career_history)
        is_pure_consulting = (total_roles > 0 and consulting_count == total_roles)

        return {
            "consulting_count": consulting_count,
            "product_count": product_count,
            "is_pure_consulting": is_pure_consulting,
            "tier1_product_experience": tier1_product,
        }

    # ─────────────────────────────────────────────
    # 8. EDUCATION TIER
    # ─────────────────────────────────────────────
    def _evaluate_education(self, education: List[Any]) -> float:
        """Returns 0-100 education score based on institution tier."""
        if not education:
            return 40.0  # Neutral

        tier_scores = {
            "tier_1": 100.0,
            "tier_2": 75.0,
            "tier_3": 55.0,
            "tier_4": 40.0,
            "unknown": 45.0,
        }

        scores = []
        for edu in education:
            tier = (edu.tier or "unknown").lower()
            scores.append(tier_scores.get(tier, 45.0))

        return max(scores)  # Best education institution wins

    # ─────────────────────────────────────────────
    # 9. BUILD TRUTHFUL DOCUMENT
    # ─────────────────────────────────────────────
    def build_truthful_document(self, candidate, verified_skills: List[str], true_persona: str) -> str:
        """Builds a semantic document using ONLY verified truths for vector search."""
        doc_parts = []

        doc_parts.append(f"True Engineering Persona: {true_persona}")
        doc_parts.append(f"Years of Experience: {candidate.profile.years_of_experience}")
        doc_parts.append(f"Reported Summary: {candidate.profile.summary}")
        doc_parts.append(f"Current Role: {candidate.profile.current_title} at {candidate.profile.current_company} ({candidate.profile.current_industry})")

        doc_parts.append("Verified Production Experience:")
        for role in candidate.career_history:
            doc_parts.append(f"- [{role.company_size}] {role.title} at {role.company} ({role.industry}, {role.duration_months}mo): {role.description}")

        if verified_skills:
            doc_parts.append(f"Verified Applied Skills: {', '.join(verified_skills)}")
        else:
            doc_parts.append("Verified Applied Skills: None explicitly found in work history.")

        # Add education context
        if candidate.education:
            edu_str = "; ".join([f"{e.degree} from {e.institution} ({e.tier or 'unknown'})" for e in candidate.education])
            doc_parts.append(f"Education: {edu_str}")

        # GitHub signal explicitly mentioned in doc for semantic retrieval
        gh = candidate.redrob_signals.github_activity_score
        if gh >= 0:
            doc_parts.append(f"GitHub Activity Score: {gh}/100")

        return "\n".join(doc_parts)

    # ─────────────────────────────────────────────
    # MASTER ENRICHMENT FUNCTION
    # ─────────────────────────────────────────────
    def enrich(self, candidate) -> Dict[str, Any]:
        """
        Master enrichment function. Returns a multi-dimensional dictionary
        with all signals needed for scoring.
        """
        # Combine all work text for lie detection
        work_text = " ".join([role.description for role in candidate.career_history if role.description])
        work_text += " " + candidate.profile.summary
        work_text += " " + " ".join([role.title for role in candidate.career_history])

        # 1. Lie Detector (Skill Verification)
        skill_verification = self._verify_skills(candidate.skills, work_text)
        trust_score = self._calculate_trust_score(
            len(skill_verification["verified"]),
            len(candidate.skills)
        )

        # 2. Derive True Persona
        true_persona = self._extract_true_persona(candidate.career_history)

        # 3. Logistics & Hard Dealbreakers (with honeypot detection)
        logistics_flags = self._evaluate_logistics_and_dealbreakers(candidate)

        # 4. Behavioral Vibe (comprehensive — all 23 signals used)
        behavioral_score = self._evaluate_behavioral_vibe(candidate.redrob_signals)

        # 5. External Validation (GitHub + platform assessments)
        external_score = self._evaluate_external_validation(candidate.redrob_signals)

        # 6. Company DNA check
        company_dna = self._evaluate_company_dna(candidate)

        # 7. Education tier
        edu_score = self._evaluate_education(candidate.education)

        # 8. Build the clean semantic document
        truthful_document = self.build_truthful_document(
            candidate, skill_verification["verified"], true_persona
        )

        return {
            "candidate_id": candidate.candidate_id,
            "vector_document": truthful_document,
            "metadata_for_scoring": {
                "trust_score": trust_score,             # 0-100
                "behavioral_score": behavioral_score,   # 0-100
                "external_score": external_score,       # 0-100 (GitHub + assessments)
                "edu_score": edu_score,                 # 0-100
                "true_persona": true_persona,
                "verified_skills": skill_verification["verified"],
                "hallucinated_skills": skill_verification["hallucinated"],
                "hallucinated_skills_count": len(skill_verification["hallucinated"]),
                "is_dealbreaker": logistics_flags["is_dealbreaker"],
                "dealbreaker_reasons": logistics_flags["reasons"],
                "is_pure_consulting": company_dna["is_pure_consulting"],
                "tier1_product_experience": company_dna["tier1_product_experience"],
                "notice_period_days": candidate.redrob_signals.notice_period_days,
                "github_activity_score": candidate.redrob_signals.github_activity_score,
            }
        }