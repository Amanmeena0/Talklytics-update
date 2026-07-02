# src/database/crud.py
"""Database CRUD operations for ConvinceSense."""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import or_, desc, asc, func
from sqlalchemy.orm import Session
from src.database import models, schemas


# ── User CRUD ─────────────────────────────────────────────────────────── #

def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    db_user = models.User(email=user.email, name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# ── Call CRUD ─────────────────────────────────────────────────────────── #

def get_call(db: Session, call_id: int) -> Optional[models.Call]:
    """Get a single call, excluding soft-deleted ones."""
    return db.query(models.Call).filter(
        models.Call.id == call_id,
        models.Call.is_deleted == False
    ).first()


def get_calls(
    db: Session,
    skip: int = 0,
    limit: int = 10,
    search: Optional[str] = None,
    is_favorite: Optional[bool] = None,
    user_id: Optional[int] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> List[models.Call]:
    """Retrieve calls with search, filter, sorting, and pagination."""
    query = db.query(models.Call).filter(models.Call.is_deleted == False)

    # Filtering by user
    if user_id is not None:
        query = query.filter(models.Call.user_id == user_id)

    # Filtering by favorite status
    if is_favorite is not None:
        query = query.filter(models.Call.is_favorite == is_favorite)

    # Searching by title or transcript content
    if search:
        search_term = f"%{search}%"
        # Join engagement_records to search transcripts
        query = query.join(models.Call.records, isouter=True).filter(
            or_(
                models.Call.title.ilike(search_term),
                models.EngagementRecord.transcript.ilike(search_term)
            )
        ).distinct()

    # Sorting
    order_col = getattr(models.Call, sort_by, models.Call.created_at)
    if sort_order.lower() == "desc":
        query = query.order_by(desc(order_col))
    else:
        query = query.order_by(asc(order_col))

    # Pagination
    return query.offset(skip).limit(limit).all()


def create_call(db: Session, call: schemas.CallCreate) -> models.Call:
    """Create a new call along with its nested records, comments, and next steps."""
    db_call = models.Call(
        title=call.title,
        user_id=call.user_id,
    )
    db.add(db_call)
    db.flush()  # Populates db_call.id

    # Add initial segment records if present
    for r in call.records:
        db_rec = models.EngagementRecord(
            call_id=db_call.id,
            timestamp=r.timestamp,
            score=r.score,
            transcript=r.transcript,
            sentiment=r.sentiment,
            buying_signals=r.buying_signals,
            hesitations=r.hesitations,
            detected_intents=r.detected_intents,
            intent_confidence=r.intent_confidence,
            recommendation=r.recommendation,
            energy=r.energy,
            confidence=r.confidence,
            speaker=r.speaker
        )
        db.add(db_rec)

    # Add comments if present
    for c in call.comments:
        db_comment = models.Comment(call_id=db_call.id, content=c.content)
        db.add(db_comment)

    # Add next steps if present
    for n in call.next_steps:
        db_next = models.NextStep(
            call_id=db_call.id,
            content=n.content,
            due_date=n.due_date,
            completed=n.completed
        )
        db.add(db_next)

    db.commit()
    db.refresh(db_call)
    return db_call


def update_call(db: Session, call_id: int, call_update: schemas.CallUpdate) -> Optional[models.Call]:
    """Update call attributes."""
    db_call = get_call(db, call_id)
    if not db_call:
        return None

    update_data = call_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_call, key, value)

    db_call.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_call)
    return db_call


def soft_delete_call(db: Session, call_id: int) -> bool:
    """Soft delete a call by setting is_deleted=True."""
    db_call = db.query(models.Call).filter(models.Call.id == call_id).first()
    if not db_call:
        return False
    db_call.is_deleted = True
    db_call.updated_at = datetime.utcnow()
    db.commit()
    return True


# ── Comments & Next Steps CRUD ────────────────────────────────────────── #

def create_comment(db: Session, call_id: int, comment: schemas.CommentCreate) -> Optional[models.Comment]:
    """Add a comment to an existing call."""
    db_call = get_call(db, call_id)
    if not db_call:
        return None
    db_comment = models.Comment(call_id=call_id, content=comment.content)
    db.add(db_comment)
    db.commit()
    db.refresh(db_comment)
    return db_comment


def update_next_step(
    db: Session,
    call_id: int,
    next_step_id: int,
    next_step_update: schemas.NextStepUpdate
) -> Optional[models.NextStep]:
    """Update a specific next step belonging to a call."""
    db_next = db.query(models.NextStep).filter(
        models.NextStep.id == next_step_id,
        models.NextStep.call_id == call_id
    ).first()
    if not db_next:
        return None

    update_data = next_step_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(db_next, key, value)
        elif key == "due_date" and "due_date" in update_data:
            db_next.due_date = None

    db_next.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_next)
    return db_next


# ── Audit Log CRUD ────────────────────────────────────────────────────── #

def create_audit_log(db: Session, log: schemas.AuditLogCreate) -> models.AuditLog:
    db_log = models.AuditLog(
        user_id=log.user_id,
        action=log.action,
        details=log.details
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log


# ── Analytics & BANT ──────────────────────────────────────────────────── #

def check_bant_compliance(records: List[models.EngagementRecord]) -> dict:
    """Evaluate Budget, Authority, Need, and Timeline mentions in transcripts/intents."""
    has_budget = False
    has_authority = False
    has_need = False
    has_timeline = False

    budget_keywords = {"price", "pricing", "budget", "cost", "fee", "charge", "expensive"}
    authority_keywords = {"decision maker", "authority", "sign-off", "approve", "approval", "manager", "boss", "director", "vp", "ceo"}
    need_keywords = {"need", "want", "problem", "pain point", "issue", "solve", "requirement", "require"}
    timeline_keywords = {"timeline", "schedule", "soon", "next week", "next month", "when", "contract", "next steps"}

    for r in records:
        text = r.transcript.lower()
        intents = [i.upper() for i in r.detected_intents] if r.detected_intents else []

        if "PRICING" in intents or any(w in text for w in budget_keywords):
            has_budget = True
        if any(w in text for w in authority_keywords):
            has_authority = True
        if "INFORMATION" in intents or any(w in text for w in need_keywords):
            has_need = True
        if "COMMITMENT" in intents or any(w in text for w in timeline_keywords):
            has_timeline = True

    return {
        "budget": has_budget,
        "authority": has_authority,
        "need": has_need,
        "timeline": has_timeline
    }


def get_analytics(db: Session) -> schemas.AnalyticsResponse:
    """Calculate aggregated metrics across all active (non-deleted) calls."""
    calls = db.query(models.Call).filter(models.Call.is_deleted == False).all()
    
    total_calls = len(calls)
    favorite_calls_count = sum(1 for c in calls if c.is_favorite)
    total_comments = db.query(func.count(models.Comment.id)).join(
        models.Call
    ).filter(models.Call.is_deleted == False).scalar() or 0
    
    pending_next_steps = db.query(func.count(models.NextStep.id)).join(
        models.Call
    ).filter(
        models.Call.is_deleted == False,
        models.NextStep.completed == False
    ).scalar() or 0

    if total_calls == 0:
        return schemas.AnalyticsResponse(
            total_calls=0,
            average_interest_score=0.0,
            bant_compliance_rate=0.0,
            bant_breakdown=schemas.BANTBreakdown(
                budget_count=0,
                authority_count=0,
                need_count=0,
                timeline_count=0
            ),
            favorite_calls_count=0,
            total_comments=0,
            pending_next_steps=0
        )

    # Calculate average interest score across all segments in active calls
    all_records = db.query(models.EngagementRecord).join(
        models.Call
    ).filter(models.Call.is_deleted == False).all()
    
    avg_score = 0.0
    if all_records:
        avg_score = round(sum(r.score for r in all_records) / len(all_records), 2)

    # Calculate BANT compliance
    budget_compliant = 0
    authority_compliant = 0
    need_compliant = 0
    timeline_compliant = 0
    total_compliance_score = 0.0

    for c in calls:
        bant = check_bant_compliance(c.records)
        
        if bant["budget"]:
            budget_compliant += 1
        if bant["authority"]:
            authority_compliant += 1
        if bant["need"]:
            need_compliant += 1
        if bant["timeline"]:
            timeline_compliant += 1

        call_score = sum(1 for v in bant.values() if v) / 4.0
        total_compliance_score += call_score

    bant_compliance_rate = round(total_compliance_score / total_calls, 4)

    return schemas.AnalyticsResponse(
        total_calls=total_calls,
        average_interest_score=avg_score,
        bant_compliance_rate=bant_compliance_rate,
        bant_breakdown=schemas.BANTBreakdown(
            budget_count=budget_compliant,
            authority_count=authority_compliant,
            need_count=need_compliant,
            timeline_count=timeline_compliant
        ),
        favorite_calls_count=favorite_calls_count,
        total_comments=total_comments,
        pending_next_steps=pending_next_steps
    )
