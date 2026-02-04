# app/schemas/claim.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

ClaimType = Literal[
    "statistic", "event_date", "quote_attribution", "causal",
    "medical_science", "policy_legal", "study_says", "biography", "other"
]
Priority = Literal["high", "medium", "low"]

class Claim(BaseModel):
    claim_id: str
    segment_id: str
    timestamp: Optional[str] = None
    claim_text: str
    quote_from_transcript: str
    claim_type: ClaimType
    entities: List[str] = Field(default_factory=list)
    check_priority: Priority = "medium"
    needs_context: List[str] = Field(default_factory=list)
