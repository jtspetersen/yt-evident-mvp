# app/schemas/verdict.py
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Rating = Literal["VERIFIED", "LIKELY TRUE", "INSUFFICIENT EVIDENCE", "CONFLICTING EVIDENCE", "LIKELY FALSE", "FALSE"]
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
