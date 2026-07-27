
from pydantic import BaseModel
from typing import List, Optional

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    id: Optional[int] = None
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: Optional[str] = "recruiter"
    permissions: Optional[dict] = {}

class UserInDB(User):
    hashed_password: str

class LoginRequest(BaseModel):
    email: str
    password: str = "google-oauth-mock"  # For Google Auth, password isn't used directly

class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    source_type: Optional[str] = "master"
    source_role_id: Optional[int] = None
    use_web_search: bool = False

class RoleCreate(BaseModel):
    name: str
    job_description: Optional[str] = None
    heyreach_campaign_id: int
    smartlead_sender_account_id: Optional[int] = 0
    smartlead_campaign_id: Optional[int] = 0
    auto_create_call_list: bool = False

class RoleUpdate(BaseModel):
    job_description: Optional[str] = None

class RoleActivationSetup(BaseModel):
    heyreach_campaign_id: int
    smartlead_sender_account_id: Optional[int] = 0
    smartlead_campaign_id: Optional[int] = 0

class AssignmentDetail(BaseModel):
    candidate_id: int
    priority: Optional[str] = "--"
    feedback: Optional[str] = ""

class CandidateAssignment(BaseModel):
    assignments: List[AssignmentDetail]

class CandidateFeedback(BaseModel):
    candidate_id: int
    priority: str
    feedback: str

class CandidateCreate(BaseModel):
    # Required set mirrors REQUIRED_IMPORT_TARGETS in backend/services/candidate_pool.py
    # so single-add and CSV-import stay consistent. `title` is stored in the
    # candidates.headline column — same slot the CSV import pipeline uses for
    # its "title" target — so there's no separate headline field here.
    first_name: str
    last_name: str
    linkedin: str
    city: str
    title: str
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None
    about: Optional[str] = None
    role_id: Optional[int] = None

class GoogleAuthRequest(BaseModel):
    token: str  # Google ID token from frontend

# Registration schemas
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    phone: Optional[str] = None

class VerifyOTPRequest(BaseModel):
    email: str
    otp_code: str

class ResendOTPRequest(BaseModel):
    email: str

class RegisterResponse(BaseModel):
    message: str
    email: str


class ChatReplyRequest(BaseModel):
    message: str
