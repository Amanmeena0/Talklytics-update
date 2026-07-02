# src/api/app.py
"""FastAPI API for ConvinceSense.
Provides REST endpoints and WebSocket real-time streams with database persistence.
"""

import asyncio
from datetime import datetime
import json
import os
import queue
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.database import models, schemas, crud
from src.database.connection import SessionLocal, engine, get_db, Base
from src.pipelines.live_pipeline import ConvinceSensePipeline
from src.features.engagement.tracker import EngagementRecord
from src.api.auth import router as auth_router
from src.auth.dependencies import require_permission


# Create all tables on startup (as a fallback/safety measure)
Base.metadata.create_all(bind=engine)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure a default user (id=1) exists in the database on startup."""
    db = SessionLocal()
    try:
        default_user = db.query(models.User).filter_by(id=1).first()
        if not default_user:
            default_user = models.User(
                id=1,
                email="salesrep@convincesense.com",
                name="Default Sales Rep"
            )
            db.add(default_user)
            db.commit()
            print("[Database Startup] Created default sales rep user (id=1)")
    except Exception as e:
        print(f"[Database Startup Error] Failed to create default user: {e}")
    finally:
        db.close()
    yield
    # Cleanup on shutdown
    if pipeline._running:
        pipeline.stop()

app = FastAPI(
    title="ConvinceSense API",
    description="REST and WebSocket APIs for ConvinceSense Conversation Intelligence",
    version="1.0.0",
    lifespan=lifespan
)

# Include Authentication Router
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])

# CORS – allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional simple API key authentication (environment variable)
API_KEY = os.getenv("CONVINCESENSE_API_KEY")

def _require_api_key(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Initialize the live pipeline (do not start it automatically on server start)
pipeline = ConvinceSensePipeline()


# ── Health & Config ───────────────────────────────────────────────────── #

@app.get("/health")
async def health(x_api_key: str = Header(None)):
    _require_api_key(x_api_key)
    return {"status": "ok"}


@app.get("/config")
async def config(x_api_key: str = Header(None)):
    _require_api_key(x_api_key)
    return {"version": "1.0.0", "framework": "ConvinceSense"}


@app.get("/session/summary")
async def session_summary(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:read")),
):
    try:
        summary_text = pipeline.get_summary()
        return {"summary": summary_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Call CRUD REST Endpoints ─────────────────────────────────────────── #

@app.get("/calls", response_model=list[schemas.Call])
def read_calls(
    skip: int = 0,
    limit: int = 10,
    search: Optional[str] = Query(None, description="Search by title or transcript content"),
    is_favorite: Optional[bool] = Query(None, description="Filter by favorite status"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    sort_by: str = Query("created_at", description="Field to sort by (created_at, title)"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:read")),
):
    return crud.get_calls(
        db,
        skip=skip,
        limit=limit,
        search=search,
        is_favorite=is_favorite,
        user_id=user_id,
        sort_by=sort_by,
        sort_order=sort_order
    )


@app.get("/calls/{call_id}", response_model=schemas.Call)
def read_call(
    call_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:read")),
):
    db_call = crud.get_call(db, call_id)
    if not db_call:
        raise HTTPException(status_code=404, detail="Call not found")
    return db_call


@app.post("/calls", response_model=schemas.Call)
def create_call(
    call: schemas.CallCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:write")),
):
    user = crud.get_user(db, call.user_id)
    if not user:
        raise HTTPException(status_code=400, detail=f"User with id {call.user_id} does not exist")
    
    db_call = crud.create_call(db, call)
    crud.create_audit_log(db, schemas.AuditLogCreate(
        user_id=call.user_id,
        event_type="CREATE_CALL",
        details={"call_id": db_call.id, "title": db_call.title}
    ))
    return db_call


@app.patch("/calls/{call_id}", response_model=schemas.Call)
def update_call(
    call_id: int,
    call_update: schemas.CallUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:write")),
):
    db_call = crud.get_call(db, call_id)
    if not db_call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    updated = crud.update_call(db, call_id, call_update)
    crud.create_audit_log(db, schemas.AuditLogCreate(
        user_id=db_call.user_id,
        event_type="UPDATE_CALL",
        details={"call_id": call_id, "updated_fields": list(call_update.model_dump(exclude_unset=True).keys())}
    ))
    return updated


@app.delete("/calls/{call_id}")
def delete_call(
    call_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:delete")),
):
    db_call = db.query(models.Call).filter(models.Call.id == call_id).first()
    if not db_call or db_call.is_deleted:
        raise HTTPException(status_code=404, detail="Call not found")
    
    success = crud.soft_delete_call(db, call_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete call")
        
    crud.create_audit_log(db, schemas.AuditLogCreate(
        user_id=db_call.user_id,
        event_type="DELETE_CALL",
        details={"call_id": call_id}
    ))
    return {"success": True, "message": "Call soft-deleted successfully"}


# ── Call Actions Endpoints ────────────────────────────────────────────── #

@app.post("/calls/{call_id}/comments", response_model=schemas.Comment)
def add_comment(
    call_id: int,
    comment: schemas.CommentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:write")),
):
    db_call = crud.get_call(db, call_id)
    if not db_call:
        raise HTTPException(status_code=404, detail="Call not found")
        
    db_comment = crud.create_comment(db, call_id, comment)
    crud.create_audit_log(db, schemas.AuditLogCreate(
        user_id=db_call.user_id,
        event_type="ADD_COMMENT",
        details={"call_id": call_id, "comment_id": db_comment.id}
    ))
    return db_comment


@app.patch("/calls/{call_id}/next-steps", response_model=schemas.NextStep)
def update_next_step(
    call_id: int,
    next_step_update: schemas.NextStepCallUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:write")),
):
    db_call = crud.get_call(db, call_id)
    if not db_call:
        raise HTTPException(status_code=404, detail="Call not found")
        
    # Extract only the fields that were set in the request (excluding id)
    update_dict = next_step_update.model_dump(exclude_unset=True)
    update_dict.pop("id", None)
    
    db_next = crud.update_next_step(
        db, 
        call_id=call_id, 
        next_step_id=next_step_update.id, 
        next_step_update=schemas.NextStepUpdate(**update_dict)
    )
    if not db_next:
        raise HTTPException(status_code=404, detail="Next step not found for this call")
        
    crud.create_audit_log(db, schemas.AuditLogCreate(
        user_id=db_call.user_id,
        event_type="UPDATE_NEXT_STEP",
        details={"call_id": call_id, "next_step_id": db_next.id}
    ))
    return db_next

@app.post("/calls/{call_id}/next-steps", response_model=schemas.NextStep)
def create_next_step(
    call_id: int,
    next_step: schemas.NextStepCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:write")),
):
    db_call = crud.get_call(db, call_id)
    if not db_call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    db_next = models.NextStep(
        call_id=call_id,
        content=next_step.content,
        due_date=next_step.due_date,
        completed=next_step.completed
    )
    db.add(db_next)
    db.commit()
    db.refresh(db_next)
    
    crud.create_audit_log(db, schemas.AuditLogCreate(
        user_id=db_call.user_id,
        event_type="ADD_NEXT_STEP",
        details={"call_id": call_id, "next_step_id": db_next.id}
    ))
    return db_next


@app.delete("/calls/{call_id}/next-steps/{next_step_id}")
def delete_next_step(
    call_id: int,
    next_step_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("calls:delete")),
):
    db_call = crud.get_call(db, call_id)
    if not db_call:
        raise HTTPException(status_code=404, detail="Call not found")
        
    db_next = db.query(models.NextStep).filter(
        models.NextStep.id == next_step_id,
        models.NextStep.call_id == call_id
    ).first()
    if not db_next:
        raise HTTPException(status_code=404, detail="Next step not found")
        
    db.delete(db_next)
    db.commit()
    
    crud.create_audit_log(db, schemas.AuditLogCreate(
        user_id=db_call.user_id,
        event_type="DELETE_NEXT_STEP",
        details={"call_id": call_id, "next_step_id": next_step_id}
    ))
    return {"success": True, "message": "Next step deleted successfully"}



# ── Analytics Endpoints ───────────────────────────────────────────────── #

@app.get("/analytics", response_model=schemas.AnalyticsResponse)
def read_analytics(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(require_permission("analytics:read")),
):
    return crud.get_analytics(db)


# ── WebSocket Real-time Stream Endpoint ───────────────────────────────── #

@app.websocket("/ws/records")
async def websocket_endpoint(
    ws: WebSocket,
    user_id: int = 1,
    save_to_db: bool = True,
    title: Optional[str] = None
):
    await ws.accept()
    
    # Generate default title if not provided
    if not title:
        title = f"Live Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Reset tracker and clear any left-over records in the output queue
    pipeline.tracker.reset()
    while not pipeline.output_q.empty():
        try:
            pipeline.output_q.get_nowait()
        except queue.Empty:
            break
            
    # Start the live pipeline on connection
    pipeline.start()
    print(f"[API WS] Client connected. Started live pipeline. save_to_db={save_to_db}, user_id={user_id}")
    
    # Task to read messages from client to detect disconnects
    async def receive_messages():
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # Task to stream records to client and persist to DB
    async def send_records():
        loop = asyncio.get_running_loop()
        db = SessionLocal() if save_to_db else None
        db_call = None
        
        try:
            if save_to_db and db:
                # Ensure the user exists
                user = db.query(models.User).filter_by(id=user_id).first()
                if not user:
                    # Fallback/auto-create if missing
                    user = models.User(id=user_id, email=f"user_{user_id}@convincesense.com", name=f"Sales Rep {user_id}")
                    db.add(user)
                    db.commit()
                    db.refresh(user)
                    
                db_call = models.Call(title=title, user_id=user_id)
                db.add(db_call)
                db.commit()
                db.refresh(db_call)
                print(f"[API WS] Created database call record (id={db_call.id})")
                await ws.send_text(json.dumps({"type": "session_info", "call_id": db_call.id}))
                
            while True:
                # Fetch next record from queue (blocking call offloaded to thread executor)
                record = await loop.run_in_executor(None, lambda: pipeline.output_q.get())
                
                # Save to database if persistence enabled
                if save_to_db and db and db_call:
                    db_rec = models.EngagementRecord(
                        call_id=db_call.id,
                        timestamp=record.timestamp,
                        score=record.score,
                        transcript=record.transcript,
                        sentiment=record.sentiment,
                        buying_signals=record.buying_signals,
                        hesitations=record.hesitations,
                        detected_intents=record.detected_intents,
                        intent_confidence=record.intent_confidence,
                        recommendation=record.recommendation,
                        energy=record.energy,
                        confidence=record.confidence,
                        speaker=record.speaker
                    )
                    db.add(db_rec)
                    db.commit()
                
                await ws.send_text(json.dumps(record.__dict__))
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[API WS] Error sending records: {e}")
        finally:
            if save_to_db and db and db_call:
                try:
                    # Generate and save LLM summary and next steps when call ends
                    records = db.query(models.EngagementRecord).filter_by(call_id=db_call.id).all()
                    if records:
                        print(f"[API WS] Generating post-call AI summary for call {db_call.id}...")
                        # Convert DB records back to schemas for summarizer
                        mapped_records = [
                            EngagementRecord(
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
                            ) for r in records
                        ]
                        summary_text = pipeline.summarizer.generate_summary(mapped_records)
                        db_call.summary = summary_text
                        db_call.updated_at = datetime.utcnow()
                        
                        # Extract next steps from records and auto-populate next_steps table
                        for rec in records:
                            if rec.recommendation:
                                db_next = models.NextStep(
                                    call_id=db_call.id,
                                    content=f"{rec.speaker}: {rec.recommendation}",
                                    completed=False
                                )
                                db.add(db_next)
                                
                        db.commit()
                        print(f"[API WS] AI summary and next steps successfully saved for call {db_call.id}")
                except Exception as summary_err:
                    print(f"[API WS] Failed to generate/save summary: {summary_err}")
                finally:
                    db.close()

    receive_task = asyncio.create_task(receive_messages())
    send_task = asyncio.create_task(send_records())
    
    try:
        await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        receive_task.cancel()
        send_task.cancel()
        pipeline.stop()
        print("[API WS] Client disconnected. Stopped live pipeline.")
