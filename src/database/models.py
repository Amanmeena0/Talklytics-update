# src/database/models.py
"""SQLAlchemy database models for ConvinceSense."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from src.database.connection import Base


class User(Base):
    """User (Sales Rep) table."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    calls = relationship("Call", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")


class Call(Base):
    """Call recording/session table."""
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_favorite = Column(Boolean, default=False, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)  # Soft-delete support
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="calls")
    records = relationship("EngagementRecord", back_populates="call", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="call", cascade="all, delete-orphan")
    next_steps = relationship("NextStep", back_populates="call", cascade="all, delete-orphan")


class EngagementRecord(Base):
    """Segment records for a sales call (transcripts, timestamps, sentiments, buying signals)."""
    __tablename__ = "engagement_records"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False)
    timestamp = Column(Float, nullable=False)  # Seconds offset in call
    score = Column(Integer, nullable=False)    # 1-5 score
    transcript = Column(Text, nullable=False)
    sentiment = Column(String(50), nullable=False)
    buying_signals = Column(JSON, default=list, nullable=False)  # List of strings, compiles to JSONB
    hesitations = Column(JSON, default=list, nullable=False)     # List of strings, compiles to JSONB
    detected_intents = Column(JSON, default=list, nullable=False)  # List of strings, compiles to JSONB
    intent_confidence = Column(Float, default=0.0, nullable=False)
    recommendation = Column(Text, nullable=True)
    energy = Column(Float, default=0.0, nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    speaker = Column(String(50), default="Unknown", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    call = relationship("Call", back_populates="records")


class Comment(Base):
    """Comment left on a sales call."""
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    call = relationship("Call", back_populates="comments")


class NextStep(Base):
    """Next action step derived from a sales call."""
    __tablename__ = "next_steps"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, ForeignKey("calls.id"), nullable=False)
    content = Column(Text, nullable=False)
    due_date = Column(DateTime, nullable=True)
    completed = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    call = relationship("Call", back_populates="next_steps")


class AuditLog(Base):
    """Audit logs for actions performed by users."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(255), nullable=False)
    details = Column(JSON, nullable=True)  # Detailed payload, compiles to JSONB
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")
