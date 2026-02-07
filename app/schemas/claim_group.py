# app/schemas/claim_group.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ClaimGroup(BaseModel):
    group_id: str
    narrative_thesis: str
    claim_ids: List[str] = Field(default_factory=list)
    rhetorical_strategy: Optional[str] = None
