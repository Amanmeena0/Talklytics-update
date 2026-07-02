# src/database/schemas.py
"""Pydantic schemas for data validation and serialization."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── Audit Log ─────────────────────────────────────────────────────────── #

class AuditLogBase(BaseModel):
    event_type: str
    details: Optional[dict] = None

class AuditLogCreate(AuditLogBase):
    user_id: Optional[int] = None

class AuditLog(AuditLogBase):
    id: int
    user_id: Optional[int]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ── User ──────────────────────────────────────────────────────────────── #

class UserBase(BaseModel):
    email: EmailStr
    name: Optional[str] = None

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ── Engagement Record ─────────────────────────────────────────────────── #

class EngagementRecordBase(BaseModel):
    timestamp: float
    score: int
    transcript: str
    sentiment: str
    buying_signals: List[str] = []
    hesitations: List[str] = []
    detected_intents: List[str] = []
    intent_confidence: float = 0.0
    recommendation: Optional[str] = None
    energy: float = 0.0
    confidence: float = 0.0
    speaker: str = "Unknown"

class EngagementRecordCreate(EngagementRecordBase):
    pass

class EngagementRecord(EngagementRecordBase):
    id: int
    call_id: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ── Comment ───────────────────────────────────────────────────────────── #

class CommentBase(BaseModel):
    content: str

class CommentCreate(CommentBase):
    pass

class Comment(CommentBase):
    id: int
    call_id: int
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ── Next Step ─────────────────────────────────────────────────────────── #

class NextStepBase(BaseModel):
    content: str
    due_date: Optional[datetime] = None
    completed: bool = False

class NextStepCreate(NextStepBase):
    pass

class NextStepUpdate(BaseModel):
    content: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: Optional[bool] = None

class NextStepCallUpdate(BaseModel):
    id: int
    content: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: Optional[bool] = None

class NextStep(NextStepBase):
    id: int
    call_id: int
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


# ── Call ──────────────────────────────────────────────────────────────── #

class CallBase(BaseModel):
    title: str

class CallCreate(CallBase):
    user_id: int
    # Optional initial data when saving a live call
    records: List[EngagementRecordCreate] = []
    comments: List[CommentCreate] = []
    next_steps: List[NextStepCreate] = []

class CallUpdate(BaseModel):
    title: Optional[str] = None
    is_favorite: Optional[bool] = None
    summary: Optional[str] = None

class Call(CallBase):
    id: int
    user_id: int
    is_favorite: bool
    is_deleted: bool
    summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    # Detailed relations returned in GET /calls/{id}
    records: List[EngagementRecord] = []
    comments: List[Comment] = []
    next_steps: List[NextStep] = []
    
    model_config = ConfigDict(from_attributes=True)


# ── Analytics Response ────────────────────────────────────────────────── #

class BANTBreakdown(BaseModel):
    budget_count: int
    authority_count: int
    need_count: int
    timeline_count: int

class AnalyticsResponse(BaseModel):
    total_calls: int
    average_interest_score: float
    bant_compliance_rate: float  # Percentage of compliance criteria met
    bant_breakdown: BANTBreakdown
    favorite_calls_count: int
    total_comments: int
    pending_next_steps: int


# ── Auth Request/Response ────────────────────────────────────────────── #

class RegisterRequest(BaseModel):
    """Request body for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: Optional[str] = None

class LoginRequest(BaseModel):
    """Request body for user login."""
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    """Response body containing the access token."""
    access_token: str
    token_type: str = "bearer"

class RefreshResponse(BaseModel):
    """Response body after token refresh."""
    access_token: str
    token_type: str = "bearer"

class MessageResponse(BaseModel):
    """Generic message response."""
    success: bool
    message: str


# ── Session ──────────────────────────────────────────────────────────── #

class SessionInfo(BaseModel):
    """Session details for device management UI."""
    id: int
    device_name: Optional[str] = None
    browser: Optional[str] = None
    operating_system: Optional[str] = None
    ip_address: Optional[str] = None
    approximate_location: Optional[str] = None
    last_active: datetime
    created_at: datetime
    is_current: bool = False
    model_config = ConfigDict(from_attributes=True)


# ── Role & Permission ────────────────────────────────────────────────── #

class PermissionSchema(BaseModel):
    """Permission details."""
    id: int
    code: str
    description: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class RoleSchema(BaseModel):
    """Role with its associated permissions."""
    id: int
    name: str
    description: Optional[str] = None
    permissions: List[PermissionSchema] = []
    model_config = ConfigDict(from_attributes=True)

class UserRoleAssign(BaseModel):
    """Request body for assigning a role to a user."""
    role_id: int


# ── User Profile (Extended) ──────────────────────────────────────────── #

class UserProfile(BaseModel):
    """Extended user profile including auth metadata."""
    id: int
    email: EmailStr
    name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    roles: List[RoleSchema] = []
    model_config = ConfigDict(from_attributes=True)


# ── Password Management ──────────────────────────────────────────────── #

class PasswordResetRequest(BaseModel):
    """Request body for initiating password reset."""
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    """Request body for confirming password reset."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)

class PasswordChangeRequest(BaseModel):
    """Request body for authenticated password change."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
