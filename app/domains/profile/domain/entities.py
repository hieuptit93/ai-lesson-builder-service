from pydantic import BaseModel


class ChildProfile(BaseModel):
    """Child profile information from user profile API."""
    child_id: str | None = None
    child_name: str | None = None
    age: int | None = None
    interests: list[str] = []
    learning_history: list[str] = []


class UserProfile(BaseModel):
    """User profile from external API."""
    user_id: str
    profile_id: str
    child: ChildProfile | None = None
    language_preference: str = "vi"
    raw_data: dict | None = None
    is_degraded: bool = False
    degraded_reason: str | None = None
