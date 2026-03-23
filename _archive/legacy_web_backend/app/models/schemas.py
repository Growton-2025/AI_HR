
from pydantic import BaseModel
from typing import List, Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class LoginRequest(BaseModel):
    email: str
    password: str = "google-oauth-mock"  # For Google Auth, password isn't used directly

class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class RoleCreate(BaseModel):
    name: str

class CandidateAssignment(BaseModel):
    candidate_ids: List[int]
    priority: Optional[str] = "--"
    feedback: Optional[str] = ""

class CandidateFeedback(BaseModel):
    candidate_id: int
    priority: str
    feedback: str
