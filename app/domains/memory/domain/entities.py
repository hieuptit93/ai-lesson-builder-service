from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class UserMemory(BaseModel):
    user_id: str
    facts: list[MemoryFact] = Field(default_factory=list)
    query_used: str = ""
    total_found: int = 0
