from schemas import Candidate
import datetime

def compute_s2_culture_stability(candidate: Candidate) -> float:
    # Avg tenure
    roles = candidate.career_history
    if not roles:
        return 0.5
    total_months = sum(r.duration_months for r in roles)
    avg_tenure = total_months / len(roles)
    
    score = 0.8 # Base
    if avg_tenure > 24:
        score += 0.2
    elif avg_tenure < 12:
        score -= 0.3
        
    return max(0.0, min(1.0, score))

def compute_s3_availability(candidate: Candidate) -> float:
    sig = candidate.redrob_signals
    # notice_period mapping: 0=1.0, 30=0.85, 60=0.6, 90=0.3
    np = sig.notice_period_days
    np_score = 1.0
    if np >= 90: np_score = 0.3
    elif np >= 60: np_score = 0.6
    elif np >= 30: np_score = 0.85
    
    # Recency mapping (days since last active)
    today = datetime.date.today()
    try:
        last_active = datetime.date.fromisoformat(sig.last_active_date)
        days_inactive = (today - last_active).days
    except:
        days_inactive = 0
        
    recency_score = max(0.0, 1.0 - (days_inactive / 365.0))
    
    open_flag = 1.0 if sig.open_to_work_flag else 0.5
    response_rate = sig.recruiter_response_rate
    
    return np_score * recency_score * open_flag * response_rate

def check_disqualifiers(candidate: Candidate, metadata: dict):
    """
    Returns (is_hard_dq, soft_multiplier, reasons)
    Note: Domain-specific DQs (Pure Researcher, Consulting Only) have been REMOVED
    from this programmatic layer and delegated to the LLM Judge / Embeddings.
    """
    reasons = []
    is_hard = False
    multiplier = 1.0
    
    # Enricher Logistics & Dealbreakers
    if metadata.get("is_dealbreaker"):
        reasons.extend(metadata.get("dealbreaker_reasons", []))
        is_hard = True
        
    # The Lie Detector Penalty
    hallucinated = metadata.get("hallucinated_skills_count", 0)
    if hallucinated > 0:
        reasons.append(f"Soft DQ: {hallucinated} Hallucinated Skills detected")
        multiplier *= (0.5 ** hallucinated) # Halve score for each fake skill
        
    # Programmatic DQ: Ghost Candidate
    sig = candidate.redrob_signals
    today = datetime.date.today()
    try:
        last_active = datetime.date.fromisoformat(sig.last_active_date)
        days_inactive = (today - last_active).days
    except:
        days_inactive = 0
        
    if days_inactive > 180 and not sig.open_to_work_flag and sig.recruiter_response_rate < 0.2:
        reasons.append("DQ-Ghost Candidate")
        multiplier *= 0.1
        
    # Programmatic DQ: Title Chaser
    if len(candidate.career_history) >= 3:
        total_months = sum(r.duration_months for r in candidate.career_history[:3])
        if total_months < 54: # 4.5 years
            reasons.append("DQ-Title Chaser")
            multiplier *= 0.7
            
    return is_hard, multiplier, reasons

def score_candidate(candidate: Candidate, semantic_score: float, metadata: dict) -> dict:
    s1 = semantic_score
    s2 = compute_s2_culture_stability(candidate)
    
    # Blend raw availability with Advanced Behavioral Vibe
    s3_raw = compute_s3_availability(candidate)
    vibe = metadata.get("behavioral_score", 100) / 100.0
    s3 = (s3_raw * 0.4) + (vibe * 0.6)
    
    s4 = metadata.get("trust_score", 50) / 100.0
    
    # Base heuristic score before LLM judge
    base_score = (s1 * 0.50) + (s2 * 0.20) + (s3 * 0.15) + (s4 * 0.15)
    
    is_hard, multiplier, dq_reasons = check_disqualifiers(candidate, metadata)
    
    final_score = 0.0 if is_hard else (base_score * multiplier)
    
    return {
        "final_score": final_score,
        "s1_semantic": s1, "s2_culture": s2, "s3_availability": s3, "s4_trust": s4,
        "is_hard_dq": is_hard,
        "dq_reasons": dq_reasons
    }
