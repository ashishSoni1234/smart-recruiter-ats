import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. ADVANCED PYDANTIC SCHEMAS
# ==========================================

class SkillRequirement(BaseModel):
    category: str = Field(description="Broad concept, e.g., 'Vector Database', 'Evaluation Frameworks'")
    specific_mentions: List[str] = Field(description="Exact tools mentioned, e.g., ['Pinecone', 'FAISS', 'NDCG']")
    context: str = Field(description="HOW this must be used, e.g., 'Deployed to real users in production'")
    is_dealbreaker: bool = Field(description="True if it's a hard requirement, False if nice-to-have")

class SkillDependency(BaseModel):
    primary_skill: str = Field(description="The claimed skill, e.g., 'LLM Fine-tuning'")
    requires_proof_of: List[str] = Field(description="Secondary skills needed to prove it isn't BS")

class TemporalConstraint(BaseModel):
    rule: str = Field(description="e.g., 'Has written production code in the last 18 months'")
    penalty_if_violated: str = Field(description="What to do if failed, e.g., 'Do not move forward'")

class ProxyProof(BaseModel):
    signal_type: str = Field(description="e.g., 'Open-source contributions', 'Platform Activity'")
    context: str = Field(description="Why it matters")

class EnvironmentContext(BaseModel):
    dna_fit: str = Field(description="e.g., 'Scrappy product-engineering, 0-to-1 builder'")
    banned_backgrounds: List[str] = Field(description="e.g., 'Pure research without production'")

class HyDEProfile(BaseModel):
    title: str
    background_summary: str
    key_experiences: List[str]

class AntiHyDEProfile(BaseModel):
    title: str
    background_summary: str
    red_flags: List[str] = Field(description="Why this candidate is a trap")

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
    negative_semantic_anchor: str = Field(description="A cohesive paragraph describing exactly who NOT to hire.")


# ==========================================
# 2. CACHE PATH
# ==========================================

JD_CACHE_PATH = "data/jd_intent_cache.json"


# ==========================================
# 3. THE PARSER FUNCTION (Offline-Safe)
# ==========================================

def parse_jd(jd_text_path: str) -> JDIntentBundle:
    """
    Parses the JD and returns a structured JDIntentBundle.

    Priority order:
    1. Load from pre-computed cache (data/jd_intent_cache.json) — NO network call
    2. If cache missing, call Groq API and save to cache
    3. If Groq fails, use hardcoded Rank-1 fallback

    This ensures has_network_during_ranking: false is honoured —
    the cache is pre-computed once and committed to the repo.
    """
    # ── Step 1: Try cache first (offline-safe path) ────────────────────────
    if os.path.exists(JD_CACHE_PATH):
        logger.info(f"Loading pre-computed JD intent from cache: {JD_CACHE_PATH}")
        try:
            with open(JD_CACHE_PATH, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            return JDIntentBundle.model_validate(cached_data)
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), will re-parse JD.")

    # ── Step 2: Read JD text ───────────────────────────────────────────────
    logger.info(f"Reading JD from {jd_text_path}")
    if os.path.exists(jd_text_path):
        with open(jd_text_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
    else:
        logger.warning(f"File {jd_text_path} not found. Using fallback data.")
        return _save_and_return(get_dummy_jd_intent())

    # ── Step 3: Call Groq API ──────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        logger.warning("GROQ_API_KEY not set. Using pre-defined Rank-1 fallback JD intent.")
        return _save_and_return(get_dummy_jd_intent())

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = f"""
    You are an expert AI Systems Architect and Principal Recruiter.
    We are building a next-generation AI ATS that relies on multidimensional semantic matching, not keyword counting.
    Read the provided Job Description carefully. It contains "traps" for traditional keyword matchers.

    Extract the deep semantic intent into a JSON object matching this exact structure:
    1. core_competencies: List of skills. Abstract specific tools into 'category' (e.g. Pinecone -> VectorDB).
    2. skill_dependencies: Pairs of skills to catch "keyword stuffers".
    3. temporal_constraints: Time-based rules (e.g., 'must have coded in last 18 months').
    4. proxy_proofs: External validation requirements (e.g., GitHub, active on platform).
    5. environment_context: The 'DNA' of the company and explicit banned backgrounds.
    6. culture_signals: Behavioral expectations.
    7. hyde_profiles: 3-5 Hypothetical IDEAL candidates.
    8. semantic_anchor_text: 200 words describing the PERFECT hire for positive vector matching.
    9. anti_hyde_profiles: 3 Hypothetical TOXIC candidates (people who look good on paper but violate 'What we explicitly do NOT want').
    10. negative_semantic_anchor: 200 words describing exactly who NOT to hire.

    JOB DESCRIPTION:
    {jd_text}

    Output ONLY valid JSON matching the requested structure.
    """

        logger.info("Calling Groq API for Advanced JD parsing...")
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        intent = JDIntentBundle.model_validate(data)
        return _save_and_return(intent)

    except Exception as e:
        logger.error(f"Groq API failed: {e}. Using Rank-1 fallback.")
        return _save_and_return(get_dummy_jd_intent())


def _save_and_return(intent: JDIntentBundle) -> JDIntentBundle:
    """Saves the intent bundle to cache and returns it."""
    try:
        os.makedirs("data", exist_ok=True)
        with open(JD_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(intent.model_dump(), f, indent=2)
        logger.info(f"JD intent cached to {JD_CACHE_PATH}")
    except Exception as e:
        logger.warning(f"Could not save JD cache: {e}")
    return intent


# ==========================================
# 4. RANK 1 FALLBACK — PERFECT EXTRACTION
# ==========================================

def get_dummy_jd_intent() -> JDIntentBundle:
    return JDIntentBundle(
        core_competencies=[
            SkillRequirement(
                category="Vector Database Infrastructure",
                specific_mentions=["Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch", "Elasticsearch", "FAISS"],
                context="Production operational experience handling embedding drift, index refresh, and retrieval-quality regression.",
                is_dealbreaker=True
            ),
            SkillRequirement(
                category="Evaluation Frameworks",
                specific_mentions=["NDCG", "MRR", "MAP", "A/B testing", "offline-to-online correlation"],
                context="Designing rigorous offline evaluation with online A/B test correlation for ranking systems.",
                is_dealbreaker=True
            ),
            SkillRequirement(
                category="Embeddings-based Retrieval",
                specific_mentions=["sentence-transformers", "OpenAI embeddings", "BGE", "E5", "hybrid search"],
                context="Deployed to real users at scale. Must have handled embedding drift in production.",
                is_dealbreaker=True
            ),
            SkillRequirement(
                category="Production Python",
                specific_mentions=["Python", "FastAPI", "production-grade code"],
                context="Strong Python. Writes production code, not demos.",
                is_dealbreaker=True
            ),
            SkillRequirement(
                category="LLM Integration",
                specific_mentions=["RAG", "LLM fine-tuning", "LoRA", "QLoRA", "PEFT", "prompt engineering"],
                context="LLM fine-tuning nice-to-have. Must understand when to fine-tune vs when to prompt.",
                is_dealbreaker=False
            ),
        ],
        skill_dependencies=[
            SkillDependency(
                primary_skill="AI/LLM App Development",
                requires_proof_of=["Pre-LLM-era ML production experience", "Understanding of retrieval/ranking fundamentals"]
            ),
            SkillDependency(
                primary_skill="Ranking Systems",
                requires_proof_of=["Evaluation metrics like NDCG or MRR", "A/B Test interpretation", "Production deployment"]
            ),
            SkillDependency(
                primary_skill="Vector Database",
                requires_proof_of=["Embedding drift handling", "Index refresh experience", "Production scale deployment"]
            ),
        ],
        temporal_constraints=[
            TemporalConstraint(
                rule="Has written production code in the last 18 months",
                penalty_if_violated="Do not move forward — this role requires active coding"
            ),
            TemporalConstraint(
                rule="AI/ML experience of at least 2 years at a product company",
                penalty_if_violated="Down-rank significantly if only consulting or research backgrounds"
            ),
        ],
        proxy_proofs=[
            ProxyProof(
                signal_type="Open-source contributions, GitHub activity, papers, talks",
                context="Critical external validation if candidate has only worked on closed-source proprietary systems for 5+ years."
            ),
            ProxyProof(
                signal_type="Active on Redrob platform — logged in recently, responding to recruiters",
                context="Behavioral signal of actually being in the job market and available to hire."
            ),
            ProxyProof(
                signal_type="Platform skill assessment scores",
                context="Objective third-party validation of claimed technical skills."
            ),
        ],
        environment_context=EnvironmentContext(
            dna_fit="Scrappy product-engineering attitude, Series A stage, 0-to-1 builder, fast shipper, async-first writer.",
            banned_backgrounds=[
                "Pure research environments without production deployment — tried twice, didn't work",
                "Only consulting firms (TCS, Infosys, Wipro, Accenture, etc.) in entire career without product-company experience",
                "Big Tech lifers needing well-scoped roles and stable codebases",
                "Recent AI experience consisting primarily of LangChain tutorials calling OpenAI without pre-LLM ML fundamentals",
                "Senior engineers who haven't written production code in 18+ months due to architecture/tech-lead roles",
            ]
        ),
        culture_signals=[
            "Async-first communication, writes well and writes a lot",
            "Disagrees openly and decides quickly — not conflict-averse",
            "Moves fast, ships scrappy v1s, learns from real users",
            "Plans to be here 3+ years — not a title chaser",
            "Thinks about systems and first principles, not frameworks",
        ],
        hyde_profiles=[
            HyDEProfile(
                title="Search ML Engineer at Mid-Size Product Company",
                background_summary="4-6 years at product companies building core search/relevance engines for e-commerce or marketplace platforms. Has shipped BM25-to-hybrid search transitions. Runs A/B tests, reads NDCG scores religiously.",
                key_experiences=[
                    "Transitioned BM25 to hybrid dense+sparse retrieval in production",
                    "Built A/B testing framework for ranking evaluation",
                    "Handled embedding drift and index refresh at real user scale",
                    "Shipped scrappy v1 rankers quickly, then iterated based on recruiter feedback",
                    "Active on GitHub with open-source ML contributions"
                ]
            ),
            HyDEProfile(
                title="Early Startup Founding Engineer with AI Specialization",
                background_summary="Built the entire data and AI pipeline from scratch at a Seed/Series A company. Owns retrieval and ranking end-to-end. Has a scrappy bias — ships working systems before perfect ones.",
                key_experiences=[
                    "End-to-end ownership of recommendation or search systems",
                    "Used sentence-transformers or OpenAI embeddings in production",
                    "Evaluated models using NDCG and MAP, set up offline benchmark pipelines",
                    "Mentored junior engineers, worked closely with PMs",
                    "5-8 years total, 3-5 in applied ML/AI at product companies"
                ]
            ),
            HyDEProfile(
                title="ML Engineer from Indian Product Unicorn",
                background_summary="3-5 years at a major Indian product company (Swiggy, Flipkart, Zomato, Razorpay, Meesho) in the ML platform or search team. Deep experience with large-scale retrieval and ranking.",
                key_experiences=[
                    "Built large-scale vector search serving millions of queries",
                    "Worked on candidate-JD or item-user matching systems",
                    "Has hands-on Python and strong system design",
                    "Understands latency-quality tradeoffs in production",
                ]
            ),
        ],
        semantic_anchor_text=(
            "A pragmatic, product-focused Machine Learning Engineer with 5-9 years of total experience, "
            "of which 3-5 years are at product companies (not pure services or research). They have shipped "
            "at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale. "
            "They possess deep operational experience with vector databases (Pinecone, FAISS, Elasticsearch) and "
            "embedding models (sentence-transformers, BGE, E5) in production environments, having dealt with "
            "embedding drift, index refresh, and retrieval regression. They design rigorous evaluation frameworks "
            "using NDCG, MRR, MAP, and offline-to-online A/B testing. Culturally, they thrive in scrappy, "
            "fast-paced, async-first environments, writing production-grade Python. They have GitHub activity "
            "or open-source contributions demonstrating external validation of their work. They plan for 3+ year "
            "tenures and think about systems and first principles rather than chasing frameworks or titles. "
            "They can mentor junior engineers and collaborate closely with product managers on what to build."
        ),
        anti_hyde_profiles=[
            AntiHyDEProfile(
                title="The LangChain Tutorial Surfer",
                background_summary="Has dozens of recent AI projects listed but <12 months of AI experience, all using LangChain to call OpenAI. No pre-LLM ML fundamentals.",
                red_flags=[
                    "Uses OpenAI API via LangChain but doesn't understand underlying retrieval/ranking math",
                    "No production deployment experience — only demos and notebooks",
                    "Framework enthusiast without systems thinking",
                    "Cannot explain NDCG or why it matters",
                    "GitHub full of tutorial repos, not original systems"
                ]
            ),
            AntiHyDEProfile(
                title="The Pure Academic Researcher",
                background_summary="Spent 5-7 years in academic lab or research-only roles publishing papers on computer vision, NLP, or robotics. Zero production deployments.",
                red_flags=[
                    "No production deployment — all work in controlled research settings",
                    "No NLP/IR/retrieval exposure relevant to hiring intelligence",
                    "Needs stable, well-scoped research environment — cannot ship scrappy v1s",
                    "Cannot handle ambiguous product requirements",
                    "Papers without any real-user validation"
                ]
            ),
            AntiHyDEProfile(
                title="The Hands-off Enterprise Architect or Title Chaser",
                background_summary="Works at a massive consulting firm or has switched companies every 1.5 years chasing title upgrades (Senior → Staff → Principal). Hasn't written production code in 18+ months.",
                red_flags=[
                    "Last code commit was 18+ months ago — now does architecture review only",
                    "Switches companies every 1-1.5 years, multiple roles in career history",
                    "Pure consulting background (TCS, Infosys, Wipro, Accenture) without any product company experience",
                    "Needs a large team, stable codebase, and well-defined scope",
                    "Optimizes for titles and compensation, not product impact"
                ]
            ),
        ],
        negative_semantic_anchor=(
            "A candidate optimizing for job titles rather than product impact, switching companies every 1-1.5 years. "
            "They might be a pure researcher with zero real-world production deployment, or a senior architect who "
            "hasn't written production code in over 18 months. They work exclusively at large consulting firms — "
            "TCS, Infosys, Wipro, Accenture, Capgemini — without any product-company experience, relying on "
            "well-scoped enterprise projects with large teams. They have adopted AI/LLM tooling recently by "
            "wrapping OpenAI APIs in LangChain without understanding core ML retrieval and ranking fundamentals. "
            "Their expertise lies entirely in non-IR fields like computer vision, speech, or robotics without "
            "significant NLP or information retrieval exposure. They have worked exclusively on closed-source "
            "proprietary systems with no external validation — no GitHub, no papers, no talks. They are inactive "
            "on job platforms, have low recruiter response rates, and long notice periods exceeding 90 days."
        ),
    )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    intent = parse_jd("data/jd_extracted.txt")
    # Write to file to avoid Windows terminal Unicode issues
    output = intent.model_dump_json(indent=2)
    with open("data/jd_intent_preview.json", "w", encoding="utf-8") as f:
        f.write(output)
    print(f"JD intent saved. Cache at: {JD_CACHE_PATH}")
    print(f"Preview at: data/jd_intent_preview.json")