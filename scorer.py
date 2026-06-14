from schemas import Candidate
import datetime

# ─────────────────────────────────────────────
# S2: CULTURE / STABILITY SCORE
# ─────────────────────────────────────────────
def compute_s2_culture_stability(candidate: Candidate, metadata: dict) -> float:
    """
    Scores cultural fit based on tenure stability, company DNA, and JD culture signals.
    Returns 0.0 - 1.0
    """
    roles = candidate.career_history
    if not roles:
        return 0.4

    # ── Average tenure ─────────────────────────────────
    total_months = sum(r.duration_months for r in roles)
    avg_tenure = total_months / len(roles)

    score = 0.6  # Neutral base
    if avg_tenure >= 30:     # 2.5+ years avg tenure — strong stability
        score += 0.25
    elif avg_tenure >= 18:   # 1.5+ years — acceptable
        score += 0.10
    elif avg_tenure < 12:    # < 1 year average — job hopper
        score -= 0.25

    # ── Company DNA (product vs consulting) ────────────
    if metadata.get("is_pure_consulting"):
        score -= 0.20   # JD explicitly bans pure consulting-only backgrounds
    if metadata.get("tier1_product_experience"):
        score += 0.10   # Bonus for Indian unicorn/top startup experience

    # ── Title Chaser check ─────────────────────────────
    # If 3+ recent jobs each lasted < 18 months → title chaser
    recent = candidate.career_history[:3]
    if len(recent) >= 3:
        recent_months = sum(r.duration_months for r in recent)
        if recent_months < 54:   # < 18 months avg over recent 3 jobs
            score -= 0.15

    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────
# S3: AVAILABILITY SCORE
# ─────────────────────────────────────────────
def compute_s3_availability(candidate: Candidate) -> float:
    """
    Scores candidate availability based on notice period, recency, and job-seeking signals.
    Returns 0.0 - 1.0
    """
    sig = candidate.redrob_signals

    # ── Notice period ──────────────────────────────────
    notice_period_days = sig.notice_period_days
    if notice_period_days <= 0:
        np_score = 1.0
    elif notice_period_days <= 15:
        np_score = 0.95
    elif notice_period_days <= 30:
        np_score = 0.85
    elif notice_period_days <= 45:
        np_score = 0.70
    elif notice_period_days <= 60:
        np_score = 0.50
    elif notice_period_days <= 90:
        np_score = 0.30
    else:
        np_score = 0.10  # >90 days (hard DQ should catch these but belt+suspenders)

    # ── Recency (last active) ──────────────────────────
    today = datetime.date.today()
    try:
        last_active = datetime.date.fromisoformat(sig.last_active_date)
        days_inactive = (today - last_active).days
    except Exception:
        days_inactive = 30  # Assume mildly inactive if date is invalid

    recency_score = max(0.0, 1.0 - (days_inactive / 180.0))  # 0 at 6 months inactive

    # ── Job-seeking signals ────────────────────────────
    open_flag = 1.0 if sig.open_to_work_flag else 0.6
    response_rate = sig.recruiter_response_rate

    # Blended availability
    availability = (np_score * 0.40) + (recency_score * 0.25) + (open_flag * 0.20) + (response_rate * 0.15)
    return max(0.0, min(1.0, availability))


# ─────────────────────────────────────────────
# DISQUALIFIER CHECKS
# ─────────────────────────────────────────────
def check_disqualifiers(candidate: Candidate, metadata: dict):
    """
    Returns (is_hard_dq, soft_multiplier, reasons)
    Hard DQ: candidate is completely excluded (score = 0)
    Soft multiplier: score penalty (0.0 - 1.0, multiplicative)
    """
    reasons = []
    is_hard = False
    multiplier = 1.0

    # ── Hard DQ from enricher (notice period >90d, honeypot) ──
    if metadata.get("is_dealbreaker"):
        reasons.extend(metadata.get("dealbreaker_reasons", []))
        is_hard = True

    # ── Soft DQ: Skill Hallucination Penalty ──────────────────
    # Fixed: Linear penalty, NOT exponential. Cap at 50% reduction.
    hallucinated_count = metadata.get("hallucinated_skills_count", 0)
    total_skills = len(candidate.skills)
    if total_skills > 0 and hallucinated_count > 0:
        hallucination_rate = hallucinated_count / total_skills
        if hallucination_rate > 0.7:
            reasons.append(f"High Hallucination: {hallucinated_count}/{total_skills} skills unverified in work history")
            multiplier *= 0.55
        elif hallucination_rate > 0.4:
            reasons.append(f"Moderate Hallucination: {hallucinated_count}/{total_skills} skills unverified")
            multiplier *= 0.75

    # ── Soft DQ: Ghost Candidate ──────────────────────────────
    # Must meet ALL three conditions to be a ghost
    sig = candidate.redrob_signals
    today = datetime.date.today()
    try:
        last_active = datetime.date.fromisoformat(sig.last_active_date)
        days_inactive = (today - last_active).days
    except Exception:
        days_inactive = 0

    if days_inactive > 180 and not sig.open_to_work_flag and sig.recruiter_response_rate < 0.2:
        reasons.append(f"Ghost Candidate: {days_inactive}d inactive, not open to work, <20% response rate")
        multiplier *= 0.20

    # ── Soft DQ: Notice Period warning (31-90 days) ───────────
    notice = sig.notice_period_days
    if 30 < notice <= 90:
        multiplier *= max(0.70, 1.0 - ((notice - 30) / 200))  # Gradual penalty

    return is_hard, multiplier, reasons


# ─────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────
def score_candidate(candidate: Candidate, semantic_score: float, metadata: dict) -> dict:
    """
    Final composite scorer with 6 weighted components.

    Components:
      S1 (50%) — Semantic match (Cross-Encoder + Hybrid Retrieval blend)
      S2 (15%) — Culture / stability fit
      S3 (12%) — Availability (notice period + recency + responsiveness)
      S4 (10%) — Trust score (verified vs claimed skills)
      S5 (08%) — External validation (GitHub + platform assessments)
      S6 (05%) — Education tier

    All components are 0.0-1.0 before weighting.
    """
    s1 = semantic_score  # 0-1, from cross-encoder + hybrid retrieval

    s2 = compute_s2_culture_stability(candidate, metadata)

    s3 = compute_s3_availability(candidate)

    # Blend raw availability with behavioral vibe
    vibe = metadata.get("behavioral_score", 60) / 100.0
    s3_blended = (s3 * 0.50) + (vibe * 0.50)

    s4 = metadata.get("trust_score", 50) / 100.0

    s5 = metadata.get("external_score", 30) / 100.0  # GitHub + skill assessments

    s6 = metadata.get("edu_score", 40) / 100.0

    # Weighted composite
    base_score = (
        (s1 * 0.50) +
        (s2 * 0.15) +
        (s3_blended * 0.12) +
        (s4 * 0.10) +
        (s5 * 0.08) +
        (s6 * 0.05)
    )

    is_hard, multiplier, dq_reasons = check_disqualifiers(candidate, metadata)

    final_score = 0.0 if is_hard else (base_score * multiplier)

    return {
        "final_score": final_score,
        "s1_semantic": s1,
        "s2_culture": s2,
        "s3_availability": s3_blended,
        "s4_trust": s4,
        "s5_external": s5,
        "s6_education": s6,
        "base_score": base_score,
        "multiplier": multiplier,
        "is_hard_dq": is_hard,
        "dq_reasons": dq_reasons,
    }
