# app/schemas/verdict.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Rating = Literal["TRUE", "LIKELY TRUE", "INSUFFICIENT EVIDENCE", "CONFLICTING EVIDENCE", "LIKELY FALSE", "FALSE"]
Severity = Literal["high", "medium", "low"]

class Citation(BaseModel):
    source_id: str
    snippet_id: str
    tier: int
    url: str
    quote: str

class Verdict(BaseModel):
    claim_id: str
    rating: Rating
    confidence: float
    explanation: str
    corrected_claim: Optional[str] = None
    severity: Severity = "medium"
    source_tiers_used: List[int] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    rhetorical_issues: List[str] = Field(default_factory=list)


NarrativeRating = Literal[
    "SUPPORTED", "PARTIALLY SUPPORTED", "MISLEADING",
    "LARGELY MISLEADING", "UNSUPPORTED"
]

class GroupVerdict(BaseModel):
    group_id: str
    narrative_thesis: str
    narrative_rating: NarrativeRating
    narrative_confidence: float
    explanation: str
    rhetorical_issues: List[str] = Field(default_factory=list)
    reasoning_gap: Optional[str] = None
    claim_ids: List[str] = Field(default_factory=list)
    individual_ratings_summary: dict = Field(default_factory=dict)
