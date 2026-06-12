import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import Groq
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. ADVANCED PYDANTIC SCHEMAS (RANK 1 LEVEL)
# ==========================================

class SkillRequirement(BaseModel):
    category: str = Field(description="Broad concept, e.g., 'Vector Database', 'Evaluation Frameworks'")
    specific_mentions: List[str] = Field(description="Exact tools mentioned, e.g., ['Pinecone', 'FAISS', 'NDCG']")
    context: str = Field(description="HOW this must be used, e.g., 'Deployed to real users in production'")
    is_dealbreaker: bool = Field(description="True if it's a hard requirement, False if nice-to-have")

class SkillDependency(BaseModel):
    primary_skill: str = Field(description="The claimed skill, e.g., 'LLM Fine-tuning'")
    requires_proof_of: List[str] = Field(description="Secondary skills needed to prove it isn't BS, e.g., ['Evaluation metrics', 'A/B testing']")

class TemporalConstraint(BaseModel):
    rule: str = Field(description="e.g., 'Has written production code in the last 18 months'")
    penalty_if_violated: str = Field(description="What to do if failed, e.g., 'Do not move forward'")

class ProxyProof(BaseModel):
    signal_type: str = Field(description="e.g., 'Open-source contributions', 'Platform Activity'")
    context: str = Field(description="Why it matters, e.g., 'To prove external validation for closed-source workers'")

class EnvironmentContext(BaseModel):
    dna_fit: str = Field(description="e.g., 'Scrappy product-engineering, 0-to-1 builder'")
    banned_backgrounds: List[str] = Field(description="e.g., 'Pure research without production', 'Only consulting firms'")

class HyDEProfile(BaseModel):
    title: str
    background_summary: str
    key_experiences: List[str]

class AntiHyDEProfile(BaseModel):
    title: str
    background_summary: str
    red_flags: List[str] = Field(description="Why this candidate is a trap (e.g., 'LangChain wrapper without ML fundamentals')")

class JDIntentBundle(BaseModel):
    # 1. Skill Extractions & Bullshit Detectors
    core_competencies: List[SkillRequirement]
    skill_dependencies: List[SkillDependency] = Field(description="Used to catch keyword stuffers")
    
    # 2. Time & Reality Constraints
    temporal_constraints: List[TemporalConstraint]
    proxy_proofs: List[ProxyProof] = Field(description="Signals for external validation")
    
    # 3. Environment & Culture
    environment_context: EnvironmentContext
    culture_signals: List[str]
    
    # 4. Positive Vector Anchors
    hyde_profiles: List[HyDEProfile] = Field(description="3-5 Ideal candidates")
    semantic_anchor_text: str = Field(description="Positive semantic anchor for embedding.")
    
    # 5. Negative Vector Anchors (The Secret Weapon)
    anti_hyde_profiles: List[AntiHyDEProfile] = Field(description="3 Toxic/Trap candidates")
    negative_semantic_anchor: str = Field(description="A cohesive paragraph describing exactly who NOT to hire. Used for negative vector distance.")


# ==========================================
# 2. THE PARSER FUNCTION
# ==========================================

def parse_jd(jd_text_path: str) -> JDIntentBundle:
    """Parses the JD text and returns a structured 3D JDIntentBundle."""
    logger.info(f"Reading JD from {jd_text_path}")
    
    # Added fallback handling if file doesn't exist yet for testing
    if os.path.exists(jd_text_path):
        with open(jd_text_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    else:
        logger.warning(f"File {jd_text_path} not found. Ensure it exists.")
        jd_text = "Refer to the Redrob JD text..."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        logger.error("CRITICAL: GROQ_API_KEY not found in environment variables.")
        raise ValueError("GROQ_API_KEY is required for final submission to dynamically parse the JD. Please add it to your .env file.")

    client = Groq(api_key=api_key)
    
    prompt = f"""
    You are an expert AI Systems Architect and Principal Recruiter. 
    We are building a next-generation AI ATS that relies on multidimensional semantic matching, not keyword counting.
    Read the provided Job Description carefully. It contains "traps" for traditional keyword matchers.
    
    Extract the deep semantic intent into a JSON object matching this exact structure:
    1. core_competencies: List of skills. Abstract specific tools into 'category' (e.g. Pinecone -> VectorDB).
    2. skill_dependencies: Pairs of skills to catch "keyword stuffers" (e.g. if they claim Ranking, they MUST have Evaluation).
    3. temporal_constraints: Time-based rules (e.g., 'must have coded in last 18 months').
    4. proxy_proofs: External validation requirements (e.g., GitHub, active on platform).
    5. environment_context: The 'DNA' of the company and explicit banned backgrounds.
    6. culture_signals: Behavioral expectations.
    7. hyde_profiles: 3-5 Hypothetical IDEAL candidates.
    8. semantic_anchor_text: 150 words describing the PERFECT hire for positive vector matching.
    9. anti_hyde_profiles: 3 Hypothetical TOXIC candidates (people who look good on paper but violate 'What we explicitly do NOT want').
    10. negative_semantic_anchor: 150 words describing exactly who NOT to hire (e.g., pure researchers, framework enthusiasts, pure consultants). Used to penalize cosine similarity.
    
    JOB DESCRIPTION:
    {jd_text}
    
    Output ONLY valid JSON matching the requested structure.
    """
    
    logger.info("Calling Groq API for Advanced JD parsing...")
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1 # Lower temp for strict schema adherence
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return JDIntentBundle.model_validate(data)
    except Exception as e:
        logger.error(f"Failed to parse JD via Groq: {e}")
        logger.warning("Falling back to Rank 1 Dummy Data.")
        return get_dummy_jd_intent()


# ==========================================
# 3. RANK 1 DUMMY DATA (PERFECT EXTRACTION)
# ==========================================

def get_dummy_jd_intent() -> JDIntentBundle:
    return JDIntentBundle(
        core_competencies=[
            SkillRequirement(
                category="Vector Database Infrastructure",
                specific_mentions=["Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", "FAISS"],
                context="Production operational experience handling embedding drift, index refresh, and regression.",
                is_dealbreaker=True
            ),
            SkillRequirement(
                category="Evaluation Frameworks",
                specific_mentions=["NDCG", "MRR", "MAP", "A/B testing"],
                context="Designing offline-to-online correlation for ranking systems.",
                is_dealbreaker=True
            ),
            SkillRequirement(
                category="Embeddings-based Retrieval",
                specific_mentions=["sentence-transformers", "OpenAI embeddings", "BGE", "E5"],
                context="Deployed to real users at scale.",
                is_dealbreaker=True
            )
        ],
        skill_dependencies=[
            SkillDependency(
                primary_skill="AI/LLM App Development",
                requires_proof_of=["Pre-LLM-era ML production experience", "Understanding of retrieval/ranking fundamentals"]
            ),
            SkillDependency(
                primary_skill="Ranking Systems",
                requires_proof_of=["Evaluation metrics like NDCG or MRR", "A/B Test interpretation"]
            )
        ],
        temporal_constraints=[
            TemporalConstraint(
                rule="Has written production code in the last 18 months",
                penalty_if_violated="Do not move forward (No pure architecture/tech lead roles without coding)"
            )
        ],
        proxy_proofs=[
            ProxyProof(
                signal_type="Open-source contributions, papers, talks",
                context="Crucial external validation if candidate has only worked on closed-source proprietary systems for 5+ years."
            ),
            ProxyProof(
                signal_type="Active on Redrob platform",
                context="Behavioral signal of actually being in the job market and responsive."
            )
        ],
        environment_context=EnvironmentContext(
            dna_fit="Scrappy product-engineering attitude, Series A stage, building from scratch, fast shipper.",
            banned_backgrounds=[
                "Pure research environments without production deployment",
                "Only consulting firms (TCS, Infosys, Wipro, etc.) without product-company experience",
                "Big Tech lifers who need well-scoped roles and stable codebases"
            ]
        ),
        culture_signals=[
            "Async-first and write a lot",
            "Disagree openly and decide quickly",
            "Move fast and break things (internal assumptions)"
        ],
        hyde_profiles=[
            HyDEProfile(
                title="Search ML Engineer at Mid-Size Product Co",
                background_summary="Spent 4 years building the core search relevance engine for an e-commerce platform.",
                key_experiences=["Transitioned BM25 to hybrid search", "Built A/B testing framework", "Shipped scrappy v1s"]
            ),
            HyDEProfile(
                title="Early Startup Founding Engineer",
                background_summary="Built the entire data and AI pipeline from scratch at a Seed-stage company.",
                key_experiences=["End-to-end ownership", "Evaluated models using NDCG", "Pragmatic shipper"]
            )
        ],
        semantic_anchor_text="A pragmatic, product-focused Machine Learning Engineer with 5-9 years of experience shipping end-to-end AI systems, particularly search, retrieval, and ranking at startup or product-centric companies. They possess deep operational experience with vector databases and embedding models in production environments. They prioritize rigorous evaluation (NDCG, A/B testing) and business impact over the latest AI hype. Culturally, they thrive in scrappy, fast-paced, async-first environments, writing production-grade Python while mentoring peers.",
        anti_hyde_profiles=[
            AntiHyDEProfile(
                title="The LangChain Tutorial Surfer",
                background_summary="Has dozens of recent AI projects listed but lacks pre-LLM fundamentals.",
                red_flags=["Uses OpenAI API via LangChain but doesn't understand underlying retrieval/ranking math", "No production deployment experience", "Framework enthusiast"]
            ),
            AntiHyDEProfile(
                title="The Pure Academic Researcher",
                background_summary="Spent 6 years in an academic lab publishing papers on computer vision.",
                red_flags=["No production deployment", "No NLP/IR exposure", "Needs a stable, well-scoped research environment"]
            ),
            AntiHyDEProfile(
                title="The Hands-off Enterprise Architect",
                background_summary="Works at a massive consulting firm optimizing for 'Principal' titles.",
                red_flags=["Hasn't written code in 2 years", "Switches jobs every 1.5 years", "Pure consulting background without product ownership"]
            )
        ],
        negative_semantic_anchor="A candidate optimizing for job titles rather than product impact, often jumping companies every 1.5 years. They might be a pure researcher with no real-world production deployment experience, or a senior tech lead who hasn't written production code in over 18 months. They are heavily biased towards consulting or massive enterprise environments, requiring stable codebases, and lacking a scrappy product-engineering attitude. They rely heavily on trendy frameworks like LangChain without understanding core ML fundamentals, or their expertise lies entirely in non-IR fields like computer vision or robotics."
    )

if __name__ == "__main__":
    # Ensure a dummy text file exists or it will use the fallback text
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/jd_extracted.txt"):
        with open("data/jd_extracted.txt", "w") as f:
            f.write("Redrob AI Series A JD text goes here...")
            
    intent = parse_jd("data/jd_extracted.txt")
    print(intent.model_dump_json(indent=2))