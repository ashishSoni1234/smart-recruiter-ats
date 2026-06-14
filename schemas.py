from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field

class CandidateProfile(BaseModel):
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str

class CareerHistoryItem(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    duration_months: int
    is_current: bool
    industry: str
    company_size: str
    description: str

class EducationItem(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: Optional[str] = None  # tier_1, tier_2, tier_3, tier_4, unknown

class SkillItem(BaseModel):
    name: str
    proficiency: str
    endorsements: int
    duration_months: Optional[int] = None

class CertificationItem(BaseModel):
    name: str
    issuer: str
    year: int

class LanguageItem(BaseModel):
    language: str
    proficiency: str

class RedrobSignals(BaseModel):
    profile_completeness_score: float
    signup_date: str
    last_active_date: str
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: float
    avg_response_time_hours: float
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_range_inr_lpa: Dict[str, float]
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: float
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: float
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

class Candidate(BaseModel):
    candidate_id: str
    profile: CandidateProfile
    career_history: List[CareerHistoryItem]
    education: List[EducationItem] = Field(default_factory=list)
    skills: List[SkillItem] = Field(default_factory=list)
    certifications: List[CertificationItem] = Field(default_factory=list)
    languages: List[LanguageItem] = Field(default_factory=list)
    redrob_signals: RedrobSignals
